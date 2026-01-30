/**
 * CC operations implemented in RMA proxy and CE.
 */
#include <assert.h>
#include <algorithm>
#include <cstring>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"
#include <functional>

typedef ncclResult_t (*NcclRmaFunc_t)(struct ncclComm*, ncclRmaWork*, cudaStream_t);

static ncclResult_t ncclRmaCollInit(struct ncclComm* comm) {
	struct ncclRmaCollState* st = &comm->rmaCollState;
  for (int i = 0; i < NCCL_RMA_COLL_MAX_STREAMS; i++) {
    if (st->rmaCollStream[i] == nullptr) {
      CUDACHECK(cudaStreamCreateWithFlags(&st->rmaCollStream[i], cudaStreamNonBlocking));
    }
    if (st->rmaCollEvent[i] == nullptr) {
      CUDACHECK(cudaEventCreateWithFlags(&st->rmaCollEvent[i], cudaEventDisableTiming));
    }
  }
  return ncclSuccess;
}

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

struct ncclRmaCollSchedule {
  // Topology info
  int rank;
  int nRanks;
  int localRank;
  int localRanks;
  int node;            // which node this rank belongs to
  int nNodes;          // total number of nodes
  int nNodesPow2;      // power of 2 >= nNodes

  // Data transfer info
  size_t eltSize;
  size_t chunkSize;

  // Valid node deltas for interNode communication (skipping delta=0)
  int* validNodeDeltas;
  int nValidNodeRounds;

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
	NCCLCHECK(ncclRmaCollInit(comm));

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
  ncclIntruQueueConstruct(&batch->proxyPutQueue);
  ncclIntruQueueConstruct(&batch->proxyWaitSignalQueue);
  ncclIntruQueueConstruct(&batch->cePutQueue);
  ncclIntruQueueConstruct(&batch->ceWaitSignalQueue);
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
  sched->node = comm->rank / comm->localRanks;
  sched->nNodes = comm->nRanks / comm->localRanks;
  sched->nNodesPow2 = pow2Up(sched->nNodes);

  // Initialize data transfer info
  sched->eltSize = ncclTypeSize(task->datatype);
  sched->chunkSize = 1ULL << 30; // 1GB

  // Compute valid node deltas (for interNode rounds, starting from delta=0)
  NCCLCHECK(ncclCalloc(&sched->validNodeDeltas, sched->nNodesPow2));
  int nodeDelta = 0;
  sched->nValidNodeRounds = 0;
  for (int nr = 0; nr < sched->nNodesPow2; nr++) {
    if (nodeDelta < sched->nNodes) {
      sched->validNodeDeltas[sched->nValidNodeRounds++] = nodeDelta;
    }
    nodeDelta = (nodeDelta + nr + 1) & (sched->nNodesPow2 - 1);
  }

  // Allocate batches (one per valid nodeRound)
  sched->nBatches = sched->nValidNodeRounds;
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

  plan->isRmaColl = true;
	ncclIntruQueueConstruct(&plan->rmaWorkBatchQueue);
  plan->rmaCollArgs = ncclMemoryStackAlloc<struct ncclRmaCollArgs>(&comm->memScoped);
  plan->rmaCollArgs->func = task->func;
  plan->rmaCollArgs->nBatches = 0;

  if (task->func == ncclFuncAlltoAllV) {
    // Prepare schedule context
    struct ncclRmaCollSchedule sched;
    NCCLCHECK(rmaCollTasksPrepare(comm, task, &sched));

    int batchIdx = 0;
    struct ncclRmaWorkBatch* curBatch = sched.batchesHead;

    // Calculate actual buffer addresses from window info
    void* sendBuff = (char*)task->sendWin->userPtr + task->sendWinOffset;

    while (curBatch != nullptr) {
      // CE Part: intraNode communication
      if (batchIdx == 0) {
        // Batch 0: nodeRound 0 (pure intraNode, all local ranks)
        // Track per-peer signal counts for the wait task (use stack alloc, no free needed)
        int* peerSignalCounts = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int* peerRanks = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int nPeersWithSignals = 0;

        for (int lr = 0; lr < sched.localRanks; lr++) {
          int sendRank = comm->localRankToRank[(sched.localRank + lr) % sched.localRanks];
          // Send part: sched.rank --> sendRank
          size_t sendCount = task->sendcounts[sched.rank * sched.nRanks + sendRank];
          if (sendCount > 0) {
            size_t sdisp = task->sdispls[sched.rank * sched.nRanks + sendRank];
            size_t rdisp = task->rdispls[sendRank * sched.nRanks + sched.rank];
            size_t totalBytes = sendCount * sched.eltSize;
            int numChunks = 1;
            if (totalBytes > sched.chunkSize) {
              numChunks = (totalBytes + sched.chunkSize - 1) / sched.chunkSize;
            }

            for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
              size_t chunkOffset = chunkIdx * sched.chunkSize;
              size_t chunkBytes = std::min(sched.chunkSize, totalBytes - chunkOffset);

              struct ncclTaskRma* cePutTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
              cePutTask->func = ncclFuncPutSignal;
              cePutTask->ctx = 0;
              cePutTask->count = chunkBytes / sched.eltSize;
              cePutTask->datatype = task->datatype;
              cePutTask->bytes = chunkBytes;
              cePutTask->srcBuff = (char*)sendBuff + sdisp * sched.eltSize + chunkOffset;
              cePutTask->peer = sendRank;
              cePutTask->peerWinOffset = task->recvWinOffset + rdisp * sched.eltSize + chunkOffset;
              cePutTask->peerWinHost = task->recvWin;
              bool isLastChunk = (chunkIdx == numChunks - 1);
              if (isLastChunk) {
                cePutTask->signalMode = NCCL_SIGNAL;
              } else {
                cePutTask->signalMode = NCCL_SIGNAL_NONE;
              }
              ncclIntruQueueEnqueue(&curBatch->cePutQueue, cePutTask);
              curBatch->nCePut++;
            }
          }

          // Recv part: sched.rank <-- recvRank
          int recvRank = comm->localRankToRank[(sched.localRank - lr + sched.localRanks) % sched.localRanks];
          size_t recvCount = task->recvcounts[sched.rank * sched.nRanks + recvRank];
          if (recvCount > 0) {
            // Record this peer and its signal count
            peerRanks[nPeersWithSignals] = recvRank;
            peerSignalCounts[nPeersWithSignals] = 1;
            nPeersWithSignals++;
          }
          break;
        }

        // Create ONE consolidated wait task for all signals in this batch
        if (nPeersWithSignals > 0) {
          struct ncclTaskRma* ceWaitTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
          ceWaitTask->func = ncclFuncWaitSignal;
          ceWaitTask->ctx = 0;
          ceWaitTask->bytes = 0;
          ceWaitTask->srcBuff = NULL;
          ceWaitTask->srcWinHost = NULL;
          ceWaitTask->peer = 0; // consolidated wait, not specific to one peer
          ceWaitTask->peerWinOffset = 0;
          ceWaitTask->peerWinHost = 0;
          ceWaitTask->signalMode = NCCL_SIGNAL;

          // Use stack alloc for peers and nsignals arrays
          ceWaitTask->npeers = nPeersWithSignals;
          ceWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, nPeersWithSignals);
          ceWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, nPeersWithSignals);
          memcpy(ceWaitTask->peers, peerRanks, nPeersWithSignals * sizeof(int));
          memcpy(ceWaitTask->nsignals, peerSignalCounts, nPeersWithSignals * sizeof(int));
          ncclIntruQueueEnqueue(&curBatch->ceWaitSignalQueue, ceWaitTask);
          curBatch->nCeWaitSignal = 1; // Only one consolidated wait per batch
        }
      } else {
        // Batch N>0: nodeRound N's phase 2+3 (cross-rail intraNode)
        int ceNodeDelta = sched.validNodeDeltas[batchIdx];
        int ceRecvNode = (sched.node - ceNodeDelta + sched.nNodes) % sched.nNodes;
        int ceRecvRankSameRail = comm->nodeRanks[ceRecvNode].localRankToRank[sched.localRank];

        // Phase 2: Data received at sameRail rank needs to be distributed locally
        size_t relayHalfBytes = 0;
        if (task->recvWinOffset > task->relayWinOffset) {
          relayHalfBytes = (task->recvWinOffset - task->relayWinOffset) / 2;
        }
        size_t relayToggleOffset = ((batchIdx - 1) & 1) ? relayHalfBytes : 0;
        char* relayBuff = (char*)task->relayWin->userPtr + task->relayWinOffset;
        for (int lr = 0; lr < sched.localRanks; lr++) {
          int destRank = comm->localRankToRank[lr];
          size_t sendCount = task->sendcounts[ceRecvRankSameRail * sched.nRanks + destRank];
          if (sendCount == 0 || lr == sched.localRank) continue;
          size_t sdisp = task->sdispls[ceRecvRankSameRail * sched.nRanks + destRank];
          size_t rdisp = task->rdispls[destRank * sched.nRanks + ceRecvRankSameRail];
          size_t totalBytes = sendCount * sched.eltSize;
          int numChunks = 1;
          if (totalBytes > sched.chunkSize) {
            numChunks = (totalBytes + sched.chunkSize - 1) / sched.chunkSize;
          }
          for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            size_t chunkOffset = chunkIdx * sched.chunkSize;
            size_t chunkBytes = std::min(sched.chunkSize, totalBytes - chunkOffset);

            struct ncclTaskRma* cePutTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
            cePutTask->func = ncclFuncPutSignal;
            cePutTask->ctx = 0;
            cePutTask->count = chunkBytes / sched.eltSize;
            cePutTask->datatype = task->datatype;
            cePutTask->bytes = chunkBytes;
            // transform from win+offset to address
            cePutTask->srcBuff = relayBuff + relayToggleOffset + sdisp * sched.eltSize + chunkOffset;
            cePutTask->peer = destRank;
            cePutTask->peerWinOffset = task->recvWinOffset + rdisp * sched.eltSize + chunkOffset;
            cePutTask->peerWinHost = task->recvWin;
            bool isLastChunk = (chunkIdx == numChunks - 1);
            if (isLastChunk) {
              cePutTask->signalMode = NCCL_SIGNAL;
            } else {
              cePutTask->signalMode = NCCL_SIGNAL_NONE;
            }
            ncclIntruQueueEnqueue(&curBatch->cePutQueue, cePutTask);
            curBatch->nCePut++;
          }
        }

        // Phase 3: Other local ranks gather data to send to this rank (wait for signals)
        int* peerSignalCounts = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int* peerRanks = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int nPeersWithSignals = 0;
        for (int lr = 0; lr < sched.localRanks; lr++) {
          int localPeerRank = comm->localRankToRank[lr];
          int remoteRank = comm->nodeRanks[ceRecvNode].localRankToRank[lr];
          size_t recvCount = task->recvcounts[remoteRank * sched.nRanks + sched.rank];
          if (recvCount > 0) {
            peerRanks[nPeersWithSignals] = localPeerRank;
            peerSignalCounts[nPeersWithSignals] = 1;
            nPeersWithSignals++;
          }
        }
        if (nPeersWithSignals > 0) {
          struct ncclTaskRma* ceWaitTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
          ceWaitTask->func = ncclFuncWaitSignal;
          ceWaitTask->ctx = 0;
          ceWaitTask->bytes = 0;
          ceWaitTask->srcBuff = NULL;
          ceWaitTask->srcWinHost = NULL;
          ceWaitTask->peer = 0; // consolidated wait, not specific to one peer
          ceWaitTask->peerWinOffset = 0;
          ceWaitTask->peerWinHost = 0;
          ceWaitTask->signalMode = NCCL_SIGNAL;

          ceWaitTask->npeers = nPeersWithSignals;
          ceWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, nPeersWithSignals);
          ceWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, nPeersWithSignals);
          memcpy(ceWaitTask->peers, peerRanks, nPeersWithSignals * sizeof(int));
          memcpy(ceWaitTask->nsignals, peerSignalCounts, nPeersWithSignals * sizeof(int));

          int totalSignals = 0;
          for (int i = 0; i < nPeersWithSignals; i++) {
            totalSignals += peerSignalCounts[i];
          }
          ceWaitTask->count = totalSignals;

          ncclIntruQueueEnqueue(&curBatch->ceWaitSignalQueue, ceWaitTask);
          curBatch->nCeWaitSignal = 1; // Only one consolidated wait per batch
        }
      }

      // Proxy Part: interNode communication for NEXT nodeRound
      int proxyNodeIdx = batchIdx + 1;
      if (proxyNodeIdx < sched.nValidNodeRounds) {
        int proxyNodeDelta = sched.validNodeDeltas[proxyNodeIdx];

        int sendNode = (sched.node + proxyNodeDelta) % sched.nNodes;
        int sendRankSameRail = comm->nodeRanks[sendNode].localRankToRank[sched.localRank];
        int recvNode = (sched.node - proxyNodeDelta + sched.nNodes) % sched.nNodes;
        int recvRankSameRail = comm->nodeRanks[recvNode].localRankToRank[sched.localRank];

        // Phase 1: recvRankSameRail --> rank (same rail, interNode)
        {
          // Create ncclTaskRma for receiving from recvRankSameRail
          // Enqueue to curBatch->proxyWaitSignalQueue
          int nSignalsFromRecvRankSameRail = 0;
          for (int lr = 0; lr < sched.localRanks; lr++) {
            int destRank = comm->nodeRanks[sched.node].localRankToRank[lr];
            size_t sendCount = task->sendcounts[recvRankSameRail * sched.nRanks + destRank];
            if (sendCount > 0) {
              nSignalsFromRecvRankSameRail++;
            }
          }
          if (nSignalsFromRecvRankSameRail > 0) {
            struct ncclTaskRma* proxyWaitTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
            proxyWaitTask->func = ncclFuncWaitSignal;
            proxyWaitTask->ctx = 0;
            proxyWaitTask->bytes = 0;
            proxyWaitTask->srcBuff = NULL;
            proxyWaitTask->srcWinHost = NULL;
            proxyWaitTask->peer = 0;
            proxyWaitTask->peerWinOffset = 0;
            proxyWaitTask->peerWinHost = NULL;
            proxyWaitTask->signalMode = NCCL_SIGNAL;

            proxyWaitTask->npeers = 1;
            proxyWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memScoped, 1);
            proxyWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memScoped, 1);
            proxyWaitTask->peers[0] = recvRankSameRail;
            proxyWaitTask->nsignals[0] = nSignalsFromRecvRankSameRail;
            proxyWaitTask->count = nSignalsFromRecvRankSameRail;
            ncclIntruQueueEnqueue(&curBatch->proxyWaitSignalQueue, proxyWaitTask);
            curBatch->nProxyWaitSignal=1; // Only one per batch
          }
        }

        // Phase 4: rank --> all ranks on sendNode (same rail, interNode)
        {
          size_t relayHalfBytes = 0;
          if (task->recvWinOffset > task->relayWinOffset) {
            relayHalfBytes = (task->recvWinOffset - task->relayWinOffset) / 2;
          }
          size_t relayToggleOffset = (batchIdx & 1) ? relayHalfBytes : 0;
          for (int lr = 0; lr < comm->nodeRanks[sendNode].localRanks; lr++) {
            int targetRank = comm->nodeRanks[sendNode].localRankToRank[lr];
            size_t sendCount = task->sendcounts[sched.rank * sched.nRanks + targetRank];
            if (sendCount > 0) {
              size_t sdisp = task->sdispls[sched.rank * sched.nRanks + targetRank];
              size_t totalBytes = sendCount * sched.eltSize;
              int numChunks = 1;
              if (totalBytes > sched.chunkSize) {
                numChunks = (totalBytes + sched.chunkSize - 1) / sched.chunkSize;
              }
              bool receiverIsSameRailRank = targetRank == sendRankSameRail;
              for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
                size_t chunkOffset = chunkIdx * sched.chunkSize;
                size_t chunkBytes = std::min(sched.chunkSize, totalBytes - chunkOffset);

                struct ncclTaskRma* proxyPutTask = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
                proxyPutTask->func = ncclFuncPutSignal;
                proxyPutTask->ctx = 0;
                proxyPutTask->count = chunkBytes / sched.eltSize;
                proxyPutTask->datatype = task->datatype;
                proxyPutTask->bytes = chunkBytes;
                proxyPutTask->srcWinOffset = task->sendWinOffset + sdisp * sched.eltSize + chunkOffset;
                proxyPutTask->srcWinHost = task->sendWin;
                proxyPutTask->peer = sendRankSameRail;
                proxyPutTask->peerWinOffset = receiverIsSameRailRank ? task->recvWinOffset +  sdisp * sched.eltSize + chunkOffset
                                              : task->relayWinOffset + relayToggleOffset + sdisp * sched.eltSize + chunkOffset;
                proxyPutTask->peerWinHost = receiverIsSameRailRank ? task->recvWin : task->relayWin;
                bool isLastChunk = (chunkIdx == numChunks - 1);
                if (isLastChunk) {
                  proxyPutTask->signalMode = NCCL_SIGNAL;
                } else {
                  proxyPutTask->signalMode = NCCL_SIGNAL_NONE;
                }
                ncclIntruQueueEnqueue(&curBatch->proxyPutQueue, proxyPutTask);
                curBatch->nProxyPut++;
              }
            }
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
    planner->nTasksRmaColl -= 1;
    plan->rmaCollArgs->nBatches = nValidBatches;
    if (sched.validNodeDeltas) free(sched.validNodeDeltas);
  }
  return ncclSuccess;
}
