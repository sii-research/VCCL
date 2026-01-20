//#include "op128.h"
#include <stdio.h>
#include "sync_kernel.h"

__device__ __forceinline__ void fence_acq_rel_sys() {
  #if __CUDA_ARCH__ >= 700
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  #else
    asm volatile("membar.sys;" ::: "memory");
  #endif
}

extern "C" __global__ void kernelFunc1(void* args) {
  volatile psmSyncCondition* syncCond = static_cast<psmSyncCondition*>(args);
  syncCond->proxyReadyEvent = 1;
  //printf("kernelFunc1: proxyReadyEvent set to %d\n", syncCond->proxyReadyEvent);
  //syncCond->proxyReadyEvent.store(1, std::memory_order_release);
  fence_acq_rel_sys();
  while (syncCond->proxyOpCount != 0) {
    // Wait for the proxy ops to be finished.
    // sched_yield();
    //nanosleep(1000); // Sleep for 1 microsecond
    //fence_acq_rel_sys();
  }
  //fence_acq_rel_sys();
  //printf("kernelFunc1: proxyOpCount is now %d\n", syncCond->proxyOpCount);
}

void asyncLaunchKernel(cudaStream_t stream, void* args) {
  //printf("asyncLaunchKernel: Launching kernelFunc1\n");
  kernelFunc1<<<1, 1, 0, stream>>>(args);
}

