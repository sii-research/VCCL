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

// helper function to launch RMA operations in RMA Collective
typedef ncclResult_t (*NcclRmaFunc_t)(struct ncclComm*, ncclRmaWork*, cudaStream_t);
template <typename SetWorkFn>
static ncclResult_t launchRmaOpHelper(struct ncclComm* comm, struct ncclRmaCollState* rmaCollState,
                    struct ncclRmaArgs* rmaArgs, cudaStream_t mainStream, int taskCount/*for batch->nOpType*/,
                    NcclRmaFunc_t func/*for Rma funcName*/, SetWorkFn setWorkField/*Lambda for setting tmpWork*/,
                    int& opCnt) {
  if (taskCount <= 0) {
    return ncclSuccess; // no need to update opCnt
  }
  ncclRmaWork tmpWork;
  // reset rmaArgs
  memset((void*)rmaArgs, 0, sizeof(struct ncclRmaArgs));
  rmaArgs->ctx = 0;
  rmaArgs->nRmaTasks = 0;
  rmaArgs->nRmaTasksProxy = 0;
  rmaArgs->nRmaTasksCe = 0;
  rmaArgs->runParallel = 1; // Default to run in parallel
  tmpWork.rmaArgs = rmaArgs;

  setWorkField(tmpWork);

  if (opCnt == 0) {
    // Launch the first work on main stream
    NCCLCHECK(func(comm, &tmpWork, mainStream));
  } else {
    // Launch the following works on rmaCollStreams
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
    // Note: one-sided host api does not support cuda graph yet.
    bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
    assert(!capturing && "RMA Collective does not support cuda graph yet.");

    cudaStream_t mainStream = comm->planner.streams->stream;
    struct ncclRmaCollState* rmaCollState = &comm->rmaCollState;
    struct ncclRmaArgs* rmaArgs = nullptr;
    NCCLCHECK(ncclCalloc(&rmaArgs, 1));

    // Iterate through each RMA work batch
    struct ncclRmaWorkBatch* batch = ncclIntruQueueHead(&plan->rmaWorkBatchQueue);
    while (batch != nullptr) {
        int opCnt = 0; // counter for number of operations launched in this batch
        // 1. Launch ProxyPut
        NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
            batch->nProxyPut,
            ncclRmaPutProxy,
            [&](ncclRmaWork& w) { w.rmaArgs->runParallel = 0; w.rmaArgs->nRmaTasksProxy = batch->nProxyPut; w.rmaTaskQueueProxy = batch->proxyPutQueue; },
            opCnt), ret, fail);
        // 2. Launch ProxyWaitSignal
        NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
            batch->nProxyWaitSignal,
            ncclRmaWaitSignalProxy,
            [&](ncclRmaWork& w) { w.rmaArgs->nRmaTasksProxy = batch->nProxyWaitSignal; w.rmaTaskQueueProxy = batch->proxyWaitSignalQueue; },
            opCnt), ret, fail);
        // 3. Launch CePut
        NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
            batch->nCePut,
            ncclRmaPutCe,
            [&](ncclRmaWork& w) { w.rmaArgs->nRmaTasksCe = batch->nCePut; w.rmaTaskQueueCe = batch->cePutQueue; },
            opCnt), ret, fail);
        // 4. Launch CeWaitSignal
        NCCLCHECKGOTO(launchRmaOpHelper(comm, rmaCollState, rmaArgs, mainStream,
            batch->nCeWaitSignal,
            ncclRmaWaitSignalCe,
            [&](ncclRmaWork& w) { w.rmaArgs->nRmaTasksCe = batch->nCeWaitSignal; w.rmaTaskQueueCe = batch->ceWaitSignalQueue; },
            opCnt), ret, fail);
        // Synchronize streams back to main stream only if they were used
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

ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  return ncclSuccess;
}
