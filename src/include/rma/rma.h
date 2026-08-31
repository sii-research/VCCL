/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_RMA_H_
#define _NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "rma/rma_ce.h"
#include "rma/rma_proxy.h"

// Internal signal mode enum
typedef enum {
  NCCL_SIGNAL_NONE = 0,        // No signaling
  NCCL_SIGNAL = 1              // Default signal operation
} ncclSignalMode_t;

struct ncclRmaArgs{
  int ctx;
  ncclFunc_t func;
  int nRmaTasks;
  int nRmaTasksProxy;
  int nRmaTasksCe;
  int runParallel; // Whether to run tasks in parallel. Only used for PutProxy.
};

struct ncclRmaState {
  struct ncclRmaProxyState rmaProxyState;
  struct ncclRmaCeState rmaCeState;
};

// Main RMA function declarations
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

// TODO: move the following contents to a new rma_coll.h file
// below are for RMA collective operations
// TODO: fill the contents!!
struct ncclRmaCollArgs {
  int nBatches;
  ncclFunc_t func;
};

// Define ncclRmaWork to eliminate confusion about RMA Plans nested inside RmaColl Plan.
// Use ncclKernelPlan internal to reuse existing rma execution infrastructure.
// Only rmaArgs, rmaTaskQueueProxy and rmaTaskQueueCe fields are used in ncclRmaWork.
using ncclRmaWork = ncclKernelPlan;

constexpr int NCCL_RMA_COLL_MAX_STREAMS = 4;
static_assert(NCCL_RMA_COLL_MAX_STREAMS >= 4, "NCCL_RMA_COLL_MAX_STREAMS must be at least 4");
struct ncclRmaCollState {
  bool initialized;
  cudaStream_t rmaCollStream[NCCL_RMA_COLL_MAX_STREAMS];
  cudaEvent_t rmaCollEvent[NCCL_RMA_COLL_MAX_STREAMS];
};

// RMA collective function declarations
ncclResult_t ncclRmaCollInit(struct ncclComm* comm);
ncclResult_t ncclRmaCollFinalize(struct ncclComm* comm);
ncclResult_t ncclLaunchRmaColl(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
#endif
