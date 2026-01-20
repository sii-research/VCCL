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

// below are for RMA collective operations
// TODO: fill the contents!!
struct ncclRmaCollArgs {
  int nBatches;
  ncclFunc_t func;
};

constexpr int NCCL_RMA_COLL_MAX_STREAMS = 4;
struct ncclRmaCollState {
  cudaStream_t rmaCollStream[NCCL_RMA_COLL_MAX_STREAMS];
  cudaEvent_t rmaCollEvent[NCCL_RMA_COLL_MAX_STREAMS];
};

// RMA collective function declarations
ncclResult_t ncclLaunchRmaColl(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
#endif
