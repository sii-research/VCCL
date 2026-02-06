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
#include "bootstrap.h"
#include "param.h"
#include "rma/rma.h"
#include <functional>

typedef ncclResult_t (*NcclRmaFunc_t)(struct ncclComm*, ncclRmaWork*, cudaStream_t);

NCCL_PARAM(RmaCollBarrier, "RMA_COLL_BARRIER", 1);
NCCL_PARAM(RmaRelayChunks, "RMA_RELAY_CHUNKS", 2);

// Helper function to dump RMA task queue
static void dumpRmaTaskQueue(const char* name,
                             struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* queue) {
  int count = 0;
  struct ncclTaskRma* task = ncclIntruQueueHead(queue);
  while (task != nullptr) {
    count++;
    task = task->next;
  }
  printf("  %s: %d\n", name, count);
  task = ncclIntruQueueHead(queue);
  while (task != nullptr) {
    printf("    Task: func=%s(%d) ctx=%d count=%zu dtype=%s bytes=%zu peer=%d signal=%d npeers=%d\n",
           ncclFuncToString(task->func), task->func, task->ctx, task->count,
           ncclDatatypeToString(task->datatype), task->bytes, task->peer,
           task->signalMode, task->npeers);
    printf("      srcBuff=%p srcWinHost=%p srcWinOffset=%zu\n",
           task->srcBuff, task->srcWinHost, task->srcWinOffset);
    printf("      peerWinHost=%p peerWinOffset=%zu\n",
           task->peerWinHost, task->peerWinOffset);
    if (task->npeers > 0 && task->peers && task->nsignals) {
      printf("      peers/nsignals:");
      for (int i = 0; i < task->npeers; i++) {
        printf(" (%d,%d)", task->peers[i], task->nsignals[i]);
      }
      printf("\n");
    }
    task = task->next;
  }
}

// Helper function to dump RMA work batch
static ncclResult_t dumpRmaWorkBatch(struct ncclRmaWorkBatch* batch, int rank) {
  if (batch == nullptr) {
    printf("RMA Work Batch: NULL\n");
    return ncclSuccess;
  }

  printf("RMA Work Batch: %p (rank=%d)\n", batch, rank);
  printf("  next: %p\n", batch->next);
  printf("  nProxyPut=%d nProxyWaitSignal=%d nCePut=%d nCeWaitSignal=%d total=%d\n",
         batch->nProxyPut, batch->nProxyWaitSignal, batch->nCePut, batch->nCeWaitSignal, batch->total);

  dumpRmaTaskQueue("proxyPutQueue", &batch->proxyPutQueue);
  dumpRmaTaskQueue("proxyWaitSignalQueue", &batch->proxyWaitSignalQueue);
  dumpRmaTaskQueue("cePutQueue", &batch->cePutQueue);
  dumpRmaTaskQueue("ceWaitSignalQueue", &batch->ceWaitSignalQueue);

  return ncclSuccess;
}


ncclResult_t ncclRmaCollInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaCollState* st = &comm->rmaCollState;
  for (int i = 0; i < NCCL_RMA_COLL_MAX_STREAMS; i++) {
    if (st->rmaCollStream[i] == nullptr) {
      CUDACHECKGOTO(cudaStreamCreateWithFlags(&st->rmaCollStream[i], cudaStreamNonBlocking), ret, fail);
    }
    if (st->rmaCollEvent[i] == nullptr) {
      CUDACHECKGOTO(cudaEventCreateWithFlags(&st->rmaCollEvent[i], cudaEventDisableTiming), ret, fail);
    }
  }
  st->initialized = true;
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaCollFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaCollState* st = &comm->rmaCollState;
  for (int i = 0; i < NCCL_RMA_COLL_MAX_STREAMS; i++) {
    if (st->rmaCollStream[i] != nullptr) {
      CUDACHECKGOTO(cudaStreamDestroy(st->rmaCollStream[i]), ret, fail);
      st->rmaCollStream[i] = nullptr;
    }
    if (st->rmaCollEvent[i] != nullptr) {
      CUDACHECKGOTO(cudaEventDestroy(st->rmaCollEvent[i]), ret, fail);
      st->rmaCollEvent[i] = nullptr;
    }
  }
  st->initialized = false;
exit:
  return ret;
fail:
  goto exit;
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

  // Relay info
  size_t relayHalfBytes; // Bytes per relay chunk
  int relayChunks;
  bool relayUseMonotonic;

  // Valid node deltas for interNode communication (skipping delta=0)
  int* validNodeDeltas;
  int nValidNodeRounds;

  // Batches
  int nBatches;
  struct ncclRmaWorkBatch* batchesHead;
};

static size_t rmaRelayOffset(const struct ncclRmaCollSchedule* sched, int batchIdx) {
  if (batchIdx < 0 || sched->relayHalfBytes == 0) return 0;
  int idx = sched->relayUseMonotonic ? batchIdx : (batchIdx % sched->relayChunks);
  return (size_t)idx * sched->relayHalfBytes;
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
  // TODO: call ncclRmaCollInit the same way as ncclRmaCeInit
  if (!rmaCollState->initialized) {
    NCCLCHECK(ncclRmaCollInit(comm));
  }

  const char* relayEnv = ncclGetEnv("NCCL_RMA_RELAY_CHUNKS");
  bool relayChunksCustom = relayEnv && relayEnv[0] != '\0';

  const char* barrierEnv = ncclGetEnv("NCCL_RMA_COLL_BARRIER");
  bool useBarrier = barrierEnv ? (ncclParamRmaCollBarrier() != 0) : !relayChunksCustom;
  if (useBarrier && comm->rank == 0) INFO(NCCL_COLL, "RMA Coll batch barrier enabled");
  else INFO(NCCL_COLL, "RMA Coll batch barrier disabled");

  int batchIdx = 0;
  // Iterate through each RMA work batch
  struct ncclRmaWorkBatch* batch = ncclIntruQueueHead(&plan->rmaWorkBatchQueue);
  while (batch != nullptr) {
    //For debugging: dump RMA work batch
    // if (comm->rank != 0) {
    //   dumpRmaWorkBatch(batch, comm->rank);
    // }
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
    if (useBarrier) {
      NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xBEEF + batchIdx), ret, fail);
    }
    batchIdx++;
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
  sched->node = comm->node;
  sched->nNodes = comm->nNodes;
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

  // Initialize relay info (per-chunk size derived from relaycounts)
  sched->relayChunks = (int)ncclParamRmaRelayChunks();
  if (sched->relayChunks < 2) {
    WARN("RMA coll: NCCL_RMA_RELAY_CHUNKS=%d is out of range [2,n]", sched->relayChunks);
    return ncclInvalidArgument;
  }
  const char* relayEnv = ncclGetEnv("NCCL_RMA_RELAY_CHUNKS");
  sched->relayUseMonotonic = relayEnv && relayEnv[0] != '\0';
  if (sched->nNodes > 1) {
    size_t relayBytes = task->relaycounts * sched->eltSize;
    if (relayBytes == 0) {
      WARN("RMA coll: relaycounts is 0 for multi-node run");
      return ncclInvalidArgument;
    }
    if (relayBytes % (size_t)sched->relayChunks != 0) {
      WARN("RMA coll: relay buffer bytes %zu not divisible by relay chunks %d", relayBytes, sched->relayChunks);
      return ncclInvalidArgument;
    }
    sched->relayHalfBytes = relayBytes / (size_t)sched->relayChunks;
    int neededRelayChunks = sched->nValidNodeRounds - 1;
    if (sched->relayUseMonotonic && sched->relayChunks < neededRelayChunks) {
      WARN("RMA coll: relay chunks %d < required %d for monotonic relay indexing",
           sched->relayChunks, neededRelayChunks);
      return ncclInvalidArgument;
    }
  } else {
    sched->relayHalfBytes = 0;
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
          ceWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memPermanent, nPeersWithSignals);
          ceWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memPermanent, nPeersWithSignals);
          memcpy(ceWaitTask->peers, peerRanks, nPeersWithSignals * sizeof(int));
          memcpy(ceWaitTask->nsignals, peerSignalCounts, nPeersWithSignals * sizeof(int));
          ncclIntruQueueEnqueue(&curBatch->ceWaitSignalQueue, ceWaitTask);
          curBatch->nCeWaitSignal = 1; // Only one consolidated wait per batch
        }
      } else {
        // Batch N>0: nodeRound N's phase 2+3 (cross-rail intraNode)
        int ceNodeDelta = sched.validNodeDeltas[batchIdx];
        int ceRecvNode = (sched.node - ceNodeDelta + sched.nNodes) % sched.nNodes;

        // Phase 2: (sched.rank ---relay--> other_local_ranks) part in the whole chain
        // (ceRecvRankSameRail ---> sched.rank ---relay--> other_local_ranks)
        // Data received at sameRail rank needs to be distributed locally
        int ceRecvRankSameRail = comm->nodeRanks[ceRecvNode].localRankToRank[sched.localRank];
        size_t relayToggleOffset = rmaRelayOffset(&sched, batchIdx - 1);
        size_t innerOffset = 0; // accumulated offset in relay buffer between multiple ranks on the same node
        for (int lr = 0; lr < sched.localRanks; lr++) {
          int destRank = comm->localRankToRank[lr];
          size_t sendCount = task->sendcounts[ceRecvRankSameRail * sched.nRanks + destRank];
          if (sendCount == 0 || lr == sched.localRank) continue;
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
            cePutTask->srcBuff = (char*)task->relayWin->userPtr + task->relayWinOffset + relayToggleOffset + innerOffset + chunkOffset;
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
          innerOffset += totalBytes;
        }

        // Phase 3: (sched.rank <---relay--- other_local_ranks) part in the whole chain
        // (sched.rank <---relay--- other_local_ranks <--- local_ranks_of_ceRecvNode)
        // Other local ranks gather data to send to this rank (wait for signals)
        int* peerSignalCounts = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int* peerRanks = ncclMemoryStackAlloc<int>(&comm->memScoped, sched.localRanks);
        int nPeersWithSignals = 0;
        for (int lr = 0; lr < sched.localRanks; lr++) {
          if (lr == sched.localRank) continue;
          int localPeerRank = comm->localRankToRank[lr];
          int srcRank = comm->nodeRanks[ceRecvNode].localRankToRank[lr];
          size_t recvCount = task->recvcounts[sched.rank * sched.nRanks + srcRank];
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
          ceWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memPermanent, nPeersWithSignals);
          ceWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memPermanent, nPeersWithSignals);
          memcpy(ceWaitTask->peers, peerRanks, nPeersWithSignals * sizeof(int));
          memcpy(ceWaitTask->nsignals, peerSignalCounts, nPeersWithSignals * sizeof(int));

          ncclIntruQueueEnqueue(&curBatch->ceWaitSignalQueue, ceWaitTask);
          curBatch->nCeWaitSignal = 1; // Only one consolidated wait per batch
        }
      }

      // Proxy Part: interNode communication for NEXT nodeRound
      int proxyNodeIdx = batchIdx + 1;
      if (proxyNodeIdx < sched.nValidNodeRounds) {
        int proxyNodeDelta = sched.validNodeDeltas[proxyNodeIdx];

        // Phase 1: sched.rank <-- recvRankSameRail (same rail, interNode)
        {
          // Create ncclTaskRma for receiving from recvRankSameRail
          // Enqueue to curBatch->proxyWaitSignalQueue
          int recvNode = (sched.node - proxyNodeDelta + sched.nNodes) % sched.nNodes;
          int recvRankSameRail = comm->nodeRanks[recvNode].localRankToRank[sched.localRank];
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
            proxyWaitTask->peers = ncclMemoryStackAlloc<int>(&comm->memPermanent, 1);
            proxyWaitTask->nsignals = ncclMemoryStackAlloc<int>(&comm->memPermanent, 1);
            proxyWaitTask->peers[0] = recvRankSameRail;
            proxyWaitTask->nsignals[0] = nSignalsFromRecvRankSameRail;
            ncclIntruQueueEnqueue(&curBatch->proxyWaitSignalQueue, proxyWaitTask);
            curBatch->nProxyWaitSignal=1; // Only one per batch
          }
        }

        // Phase 4: sched.rank --> all ranks on sendNode (same rail, interNode)
        {
          int sendNode = (sched.node + proxyNodeDelta) % sched.nNodes;
          int sendRankSameRail = comm->nodeRanks[sendNode].localRankToRank[sched.localRank];
          size_t relayToggleOffset = rmaRelayOffset(&sched, batchIdx);
          size_t innerOffset = 0; // accumulated offset in relay buffer between multiple ranks on the same node
          for (int lr = 0; lr < comm->nodeRanks[sendNode].localRanks; lr++) {
            int targetRank = comm->nodeRanks[sendNode].localRankToRank[lr];
            size_t sendCount = task->sendcounts[sched.rank * sched.nRanks + targetRank];
            if (sendCount > 0) {
              size_t sdisp = task->sdispls[sched.rank * sched.nRanks + targetRank];
              size_t rdisp = task->rdispls[targetRank * sched.nRanks + sched.rank];
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
                proxyPutTask->peerWinOffset = receiverIsSameRailRank ? task->recvWinOffset +  rdisp * sched.eltSize + chunkOffset
                                              : task->relayWinOffset + relayToggleOffset + innerOffset + chunkOffset;
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
              if (!receiverIsSameRailRank) {
                innerOffset += totalBytes;
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
      struct ncclRmaWorkBatch* next = curBatch->next;
      if (curBatch->total > 0) {
        ncclIntruQueueEnqueue(&plan->rmaWorkBatchQueue, curBatch);
        nValidBatches++;
      }
      curBatch = next;
    }
    planner->nTasksRmaColl -= 1;
    plan->rmaCollArgs->nBatches = nValidBatches;
    if (sched.validNodeDeltas) free(sched.validNodeDeltas);
  }
  return ncclSuccess;
}
