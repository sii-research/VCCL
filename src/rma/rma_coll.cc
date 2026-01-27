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

// Helper function to check if two ranks are on same rail (same local rank, different nodes)
static inline bool isSameRail(struct ncclComm* comm, int rank1, int rank2) {
  if (comm->rankToNode[rank1] == comm->rankToNode[rank2]) {
    return false; // Same node, not inter-node
  }
  return comm->rankToLocalRank[rank1] == comm->rankToLocalRank[rank2];
}

// Helper to allocate and initialize a new RMA work batch
static ncclResult_t allocRmaWorkBatch(struct ncclRmaWorkBatch** batchOut) {
  struct ncclRmaWorkBatch* batch;
  NCCLCHECK(ncclCalloc(&batch, 1));
  batch->next = nullptr;
  batch->nProxyPut = 0;
  batch->nProxyWaitSignal = 0;
  batch->nCePut = 0;
  batch->nCeWaitSignal = 0;
  *batchOut = batch;
  return ncclSuccess;
}

ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskRmaColl* task = ncclIntruQueueDequeue(&planner->collRmaTaskQueue);

  planner->isRmaColl = true;
  plan->rmaCollArgs->func = task->func;
  plan->rmaCollArgs->nBatches = 0;

  int groupSize = comm->p2pSchedGroupSize;
  int group = comm->localRank / groupSize;
  int nGroups = comm->nRanks / groupSize;
  int nGroupsPow2 = pow2Up(nGroups);
  int rank = comm->rank;
  const size_t eltSize = ncclTypeSize(task->datatype);
  const size_t chunkSize = 1ULL << 30; // 1GB

  if (task->func == ncclFuncAlltoAllV) {
    // Collect valid groupRounds and their deltas
    int* validGroupDeltas;
    int nValidGroupRounds = 0;
    int groupDelta = 1;

    NCCLCHECK(ncclCalloc(&validGroupDeltas, nGroupsPow2));
    int groupDelta = 1;
    int nValidGroupRounds = 0;
    for (int gr = 0; gr < nGroupsPow2; gr++) {
      if (groupDelta < nGroups) {
        validGroupDeltas[nValidGroupRounds++] = groupDelta;
      }
      groupDelta = (groupDelta + gr + 1) & (nGroupsPow2 - 1);
    }

    int nBatches = nValidGroupRounds;
    struct ncclRmaWorkBatch** batches;
    NCCLCHECK(ncclCalloc(&batches, nBatches));
    for (int i = 0; i < nBatches; i++) {
      NCCLCHECK(allocRmaWorkBatch(&batches[i]));
    }

    for (int batchIdx = 0; batchIdx < nBatches; batchIdx++) {
      struct ncclRmaWorkBatch* curBatch = batches[batchIdx];
      if (batchIdx == 0) {
        for (int round = 0; round < comm->localRanks; round++) {
          int sendRank = comm->p2pSchedule[round].sendRank;
          int recvRank = comm->p2pSchedule[round].recvRank;
          if (rank == sendRank) {
            // selfcopy
          }

          size_t sendCount = task->sendcounts[rank * comm->nRanks + sendRank];
          if (sendCount > 0) {
            size_t sdisp = task->sdispls[rank * comm->nRanks + sendRank];
            size_t rdisp = task->rdispls[sendRank * comm->nRanks + rank];
            // TODO: Create ncclTaskRma and enqueue to curBatch->cePutQueue
            // TODO: Create ncclTaskRma for signal and enqueue to curBatch->ceWaitSignalQueue
            curBatch->nCePut++;
            curBatch->nCeWaitSignal++;
          }
        }
      } else {
        // groupRound > 0: Phase 2 & 3 (intraNode cross-rail)
        // This batch processes CE for groupRound[batchIdx]
        int ceGroupDelta = validGroupDeltas[batchIdx];
        int ceRecvNode = (group - ceGroupDelta + nGroups) % nGroups;
        int ceRecvRankSameRail = comm->nodeRanks[ceRecvNode].localRankToRank[comm->localRank];

        // Phase 2: Data received at sameRail rank needs to be distributed locally
        // Phase 3: Other local ranks gather data to send to this rank
        for (int lr = 0; lr < comm->localRanks; lr++) {
          int localPeerRank = comm->localRankToRank[lr];
          if (localPeerRank == rank) continue; // skip self

          // TODO: Create ncclTaskRma for phase 2 (local distribution from recvRankSameRail)
          // TODO: Create ncclTaskRma for phase 3 (local gathering to rank)
          // Enqueue to curBatch->cePutQueue and curBatch->ceWaitSignalQueue
          curBatch->nCePut++;
          curBatch->nCeWaitSignal++;
        }
      }

      // ======================================================================
      // Proxy Part: interNode communication for NEXT groupRound
      // - Batch N processes Proxy for groupRound[N+1]
      // ======================================================================
      int proxyGroupIdx = batchIdx + 1;  // Proxy ops come from next groupRound
      if (proxyGroupIdx < nValidGroupRounds) {
        int proxyGroupDelta = validGroupDeltas[proxyGroupIdx];

        int sendNode = (group + proxyGroupDelta) % nGroups;
        int sendRankSameRail = comm->nodeRanks[sendNode].localRankToRank[comm->localRank];
        int recvNode = (group - proxyGroupDelta + nGroups) % nGroups;
        int recvRankSameRail = comm->nodeRanks[recvNode].localRankToRank[comm->localRank];

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
    }

    // Link batches into plan's rmaWorkBatchQueue
    for (int i = 0; i < nBatches; i++) {
      ncclIntruQueueEnqueue(&plan->rmaWorkBatchQueue, batches[i]);
    }
    plan->rmaCollArgs->nBatches = nBatches;

    free(validGroupDeltas);
    free(batches);
  }

  return ncclSuccess;
}
