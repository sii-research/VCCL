#ifndef NCCL_SYNC_KERNEL_H_
#define NCCL_SYNC_KERNEL_H_
#include <cuda.h>
#include <cuda_runtime.h>

struct psmSyncCondition {
  int proxyOpCount;    // The total number of proxyOp's that have been enqueued in this plan.
  int proxyReadyEvent; // Event that proxy thread queries for starting progresssing.
};

// Entry point for sync kernel launch
void asyncLaunchKernel(cudaStream_t stream, void* args);

#endif // NCCL_SYNC_KERNEL_H_