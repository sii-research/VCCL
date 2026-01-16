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

// A RMA work batch is a batch of RMA operations that are executed in full parallelism.
// A ncclTaskRmaColl will be split into multiple work batches if there are too many RMA operations to fit into one batch.
// All ncclTaskRma inside a work batch are executed in parallel if possible.
// The ncclRmaWorkBatch of the same ncclTaskRmaColl run in serial.
struct ncclRmaWorkBatch {
  struct ncclRmaWorkBatch* next;
  int nProxyPut; // number of ncclTaskRma elements in proxyPutQueue
  int nProxyWaitSignal;
  int nCePut;
  int nCeWaitSignal;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> proxyPutQueue; // PutSignal & Signal Func
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> proxyWaitSignalQueue;
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> cePutQueue; // PutSignal & Signal Func
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next> ceWaitSignalQueue;
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
