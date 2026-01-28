/**
 * CC operations implemented in RMA proxy and CE.
 */
#include <assert.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"
#include <functional>

typedef ncclResult_t (*NcclRmaFunc_t)(struct ncclComm*, ncclRmaWork*, cudaStream_t);

// Helper function to launch RMA operations with proper stream management
// - If opCnt == 0: launch on mainStream (first operation)
// - If opCnt > 0: launch on rmaCollStream with event synchronization
template <typename SetWorkFn>
static ncclResult_t launchRmaOpHelper(struct ncclComm* comm, struct ncclRmaCollState* rmaCollState,
                    struct ncclRmaArgs* rmaArgs, cudaStream_t mainStream, int taskCount/*tasks of particular type*/,
                    NcclRmaFunc_t func/*Rma funcName*/, SetWorkFn setWorkField/*Lambda for setting tmpWork*/,
                    int& opCnt) {
  if (taskCount <= 0) {
    return ncclSuccess; // no need to update opCnt
  }

  ncclRmaWork tmpWork;
  // Reset rmaArgs structure
  memset((void*)rmaArgs, 0, sizeof(struct ncclRmaArgs));
  rmaArgs->ctx = 0;
  rmaArgs->nRmaTasks = 0;
  rmaArgs->nRmaTasksProxy = 0;
  rmaArgs->nRmaTasksCe = 0;
  rmaArgs->runParallel = 1;  // Default to parallel execution

  tmpWork.rmaArgs = rmaArgs;
  setWorkField(tmpWork);

  if (opCnt == 0) {
    // First operation: launch on main stream
    NCCLCHECK(func(comm, &tmpWork, mainStream));
  } else {
    // Subsequent operations: launch on separate rmaCollStream with synchronization
    cudaStream_t opStream = rmaCollState->rmaCollStream[opCnt - 1];
    cudaEvent_t opEvent = rmaCollState->rmaCollEvent[opCnt - 1];
    CUDACHECK(cudaEventRecord(opEvent, mainStream));
    CUDACHECK(cudaStreamWaitEvent(opStream, opEvent, 0));
    NCCLCHECK(func(comm, &tmpWork, opStream));
  }
  opCnt++;
  return ncclSuccess;
}

#define NCCL_NODE_SIZE 8

struct ncclRmaCollSchedule {
  // Topology info
  int rank;
  int nRanks;
  int localRank;
  int localRanks;
  int group;           // which node group this rank belongs to
  int nGroups;         // total number of node groups
  int nGroupsPow2;     // power of 2 >= nGroups

  // Data transfer info
  size_t eltSize;
  size_t chunkSize;

  // Valid group deltas for interNode communication (skipping delta=0)
  int* validGroupDeltas;
  int nValidGroupRounds;

  // Batches
  int nBatches;
  struct ncclRmaWorkBatch* batchesHead;
};

ncclResult_t ncclLaunchRmaColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;

  // Note: one-sided host api does not support cuda graph yet
  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  assert(!capturing && "RMA Collective does not support cuda graph yet.");

  cudaStream_t mainStream = comm->planner.streams->stream;
  struct ncclRmaCollState* rmaCollState = &comm->rmaCollState;
  struct ncclRmaArgs* rmaArgs = nullptr;
  NCCLCHECK(ncclCalloc(&rmaArgs, 1));

  // Iterate through each RMA work batch
  struct ncclRmaWorkBatch* batch = ncclIntruQueueHead(&plan->rmaWorkBatchQueue);
  while (batch != nullptr) {
    int opCnt = 0;  // Counter for number of operations launched in this batch

    // Launch the four types of RMA operations in parallel:
    // 1. ProxyPut
    NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
      batch->nProxyPut,
      ncclRmaPutProxy,
      [&](ncclRmaWork& w) {
        w.rmaArgs->runParallel = 0; // rmaTasks in ProxyPut are run sequentially
        w.rmaArgs->nRmaTasksProxy = batch->nProxyPut;
        w.rmaTaskQueueProxy = batch->proxyPutQueue;
      },
      opCnt), ret, fail);

    // 2. ProxyWaitSignal
    NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
      batch->nProxyWaitSignal,
      ncclRmaWaitSignalProxy,
      [&](ncclRmaWork& w) {
        w.rmaArgs->nRmaTasksProxy = batch->nProxyWaitSignal;
        w.rmaTaskQueueProxy = batch->proxyWaitSignalQueue;
      },
      opCnt), ret, fail);

    // 3. CePut
    NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
      batch->nCePut,
      ncclRmaPutCe,
      [&](ncclRmaWork& w) {
        w.rmaArgs->nRmaTasksCe = batch->nCePut;
        w.rmaTaskQueueCe = batch->cePutQueue;
      },
      opCnt), ret, fail);

    // 4. CeWaitSignal
    NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
      batch->nCeWaitSignal,
      ncclRmaWaitSignalCe,
      [&](ncclRmaWork& w) {
        w.rmaArgs->nRmaTasksCe = batch->nCeWaitSignal;
        w.rmaTaskQueueCe = batch->ceWaitSignalQueue;
      },
      opCnt), ret, fail);

    // Synchronize all secondary streams back to main stream
    for (int idx = 0; idx < opCnt - 1; idx++) {
      cudaStream_t workStream = rmaCollState->rmaCollStream[idx];
      cudaEvent_t workEvent = rmaCollState->rmaCollEvent[idx];
      CUDACHECKGOTO(cudaEventRecord(workEvent, workStream), ret, fail);
      CUDACHECKGOTO(cudaStreamWaitEvent(mainStream, workEvent, 0), ret, fail);
    }

    // Move to next batch
    batch = batch->next;
  }

exit:
  if (rmaArgs) free(rmaArgs);
  return ret;
fail:
  goto exit;
}

// Helper to allocate and initialize a new RMA work batch
static ncclResult_t allocRmaWorkBatch(struct ncclComm* comm, struct ncclRmaWorkBatch** batchOut) {
  struct ncclRmaWorkBatch* batch = ncclMemoryPoolAlloc<struct ncclRmaWorkBatch>(&comm->memPool_ncclRmaWorkBatch, &comm->memPermanent);
  batch->next = nullptr;
  batch->nProxyPut = 0;
  batch->nProxyWaitSignal = 0;
  batch->nCePut = 0;
  batch->nCeWaitSignal = 0;
  batch->total = 0;
  *batchOut = batch;
  return ncclSuccess;
}

static ncclResult_t rmaCollTasksPrepare(
    struct ncclComm* comm,
    struct ncclTaskRmaColl* task,
    struct ncclRmaCollSchedule* sched) {
  // Initialize topology info
  sched->rank = comm->rank;
  sched->nRanks = comm->nRanks;
  sched->localRank = comm->localRank;
  sched->localRanks = comm->localRanks;
  sched->group = comm->localRank / NCCL_NODE_SIZE;
  sched->nGroups = comm->nRanks / NCCL_NODE_SIZE;
  sched->nGroupsPow2 = pow2Up(sched->nGroups);

  // Initialize data transfer info
  sched->eltSize = ncclTypeSize(task->datatype);
  sched->chunkSize = 1ULL << 30; // 1GB

  // Compute valid group deltas (for interNode rounds, starting from delta=1)
  NCCLCHECK(ncclCalloc(&sched->validGroupDeltas, sched->nGroupsPow2));
  int groupDelta = 1;
  sched->nValidGroupRounds = 0;
  for (int gr = 0; gr < sched->nGroupsPow2; gr++) {
    if (groupDelta < sched->nGroups) {
      sched->validGroupDeltas[sched->nValidGroupRounds++] = groupDelta;
    }
    groupDelta = (groupDelta + gr + 1) & (sched->nGroupsPow2 - 1);
  }

  // Allocate batches (one per valid groupRound)
  sched->nBatches = sched->nValidGroupRounds;
  sched->batchesHead = nullptr;
  struct ncclRmaWorkBatch* prevBatch = nullptr;
  for (int i = 0; i < sched->nBatches; i++) {
    struct ncclRmaWorkBatch* batch;
    NCCLCHECK(allocRmaWorkBatch(comm, &batch));
    if (prevBatch == nullptr) {
      sched->batchesHead = batch;
    } else {
      prevBatch->next = batch;
    }
    prevBatch = batch;
  }

  return ncclSuccess;
}

ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskRmaColl* task = ncclIntruQueueDequeue(&planner->collRmaTaskQueue);

  planner->isRmaColl = true;
  plan->rmaCollArgs->func = task->func;
  plan->rmaCollArgs->nBatches = 0;

  if (task->func == ncclFuncAlltoAllV) {
    // Prepare schedule context
    struct ncclRmaCollSchedule sched;
    NCCLCHECK(rmaCollTasksPrepare(comm, task, &sched));

    int batchIdx = 0;
    struct ncclRmaWorkBatch* curBatch = sched.batchesHead;
    while (curBatch != nullptr) {

      // ==================================================================
      // CE Part: intraNode communication
      // ==================================================================
      if (batchIdx == 0) {
        // Batch 0: groupRound 0 (pure intraNode, all local ranks)
        for (int round = 0; round < sched.localRanks; round++) {
          int sendRank = comm->p2pSchedule[round].sendRank;
          int recvRank = comm->p2pSchedule[round].recvRank;
          if (sched.rank == sendRank) {
            // selfcopy
          }

          size_t sendCount = task->sendcounts[sched.rank * sched.nRanks + sendRank];
          if (sendCount > 0) {
            size_t sdisp = task->sdispls[sched.rank * sched.nRanks + sendRank];
            size_t rdisp = task->rdispls[sendRank * sched.nRanks + sched.rank];
            // TODO: Create ncclTaskRma and enqueue to curBatch->cePutQueue
            // TODO: Create ncclTaskRma for signal and enqueue to curBatch->ceWaitSignalQueue
            curBatch->nCePut++;
            curBatch->nCeWaitSignal++;
          }
        }
      } else {
        // Batch N>0: groupRound N's phase 2+3 (cross-rail intraNode)
        int ceGroupDelta = sched.validGroupDeltas[batchIdx];
        int ceRecvNode = (sched.group - ceGroupDelta + sched.nGroups) % sched.nGroups;
        int ceRecvRankSameRail = comm->nodeRanks[ceRecvNode].localRankToRank[sched.localRank];

        // Phase 2: Data received at sameRail rank needs to be distributed locally
        // Phase 3: Other local ranks gather data to send to this rank
        for (int lr = 0; lr < sched.localRanks; lr++) {
          int localPeerRank = comm->localRankToRank[lr];
          if (localPeerRank == sched.rank) continue; // skip self

          // TODO: Create ncclTaskRma for phase 2 (local distribution from recvRankSameRail)
          // TODO: Create ncclTaskRma for phase 3 (local gathering to rank)
          // Enqueue to curBatch->cePutQueue and curBatch->ceWaitSignalQueue
          curBatch->nCePut++;
          curBatch->nCeWaitSignal++;
        }
      }

      // ==================================================================
      // Proxy Part: interNode communication for NEXT groupRound
      // ==================================================================
      int proxyGroupIdx = batchIdx + 1;
      if (proxyGroupIdx < sched.nValidGroupRounds) {
        int proxyGroupDelta = sched.validGroupDeltas[proxyGroupIdx];

        int sendNode = (sched.group + proxyGroupDelta) % sched.nGroups;
        int sendRankSameRail = comm->nodeRanks[sendNode].localRankToRank[sched.localRank];
        int recvNode = (sched.group - proxyGroupDelta + sched.nGroups) % sched.nGroups;
        int recvRankSameRail = comm->nodeRanks[recvNode].localRankToRank[sched.localRank];

        // Phase 1: recvRankSameRail --> rank (same rail, interNode)
        {
          // TODO: Create ncclTaskRma for receiving from recvRankSameRail
          // Enqueue to curBatch->proxyWaitSignalQueue
          curBatch->nProxyWaitSignal++;
        }

        // Phase 4: rank --> all ranks on sendNode (same rail, interNode)
        {
          for (int lr = 0; lr < comm->nodeRanks[sendNode].localRanks; lr++) {
            int targetRank = comm->nodeRanks[sendNode].localRankToRank[lr];
            // TODO: Create ncclTaskRma for sending to targetRank
            // Enqueue to curBatch->proxyPutQueue
            curBatch->nProxyPut++;
          }
        }
      }
      
      curBatch->total = curBatch->nProxyPut + curBatch->nProxyWaitSignal + 
                       curBatch->nCePut + curBatch->nCeWaitSignal;
      // Move to next batch
      curBatch = curBatch->next;
      batchIdx++;
    }

    // Link batches into plan's rmaWorkBatchQueue
    curBatch = sched.batchesHead;
    int nValidBatches = 0;
    while (curBatch != nullptr) {
      if (curBatch->total > 0) {
        ncclIntruQueueEnqueue(&plan->rmaWorkBatchQueue, curBatch);
        nValidBatches++;
      }
      curBatch = curBatch->next;
    }
    plan->rmaCollArgs->nBatches = nValidBatches;
    if (sched->validGroupDeltas) free(sched->validGroupDeltas);
  }
  return ncclSuccess;
}
