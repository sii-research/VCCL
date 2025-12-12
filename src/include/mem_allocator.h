#pragma once

#include <cstdio>
#include <algorithm>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "nccl.h"

class MemAllocator {
public:
  MemAllocator();
  ~MemAllocator();

  ncclResult_t cuCallocAsync(void **ptr, CUmemGenericAllocationHandle *handlep,
                             CUmemAllocationHandleType type, size_t numBytes);

  size_t getUsedMem() const { return totalMemAllocated_; }

  bool ownsPointer(const void* ptr) const;

  ncclResult_t releaseMem(const void *ptr);

private:
  ncclResult_t computeCumemGranualirity();
  ncclResult_t allocateMem(void **ptr, CUmemGenericAllocationHandle *handlep,
                           CUmemAllocationHandleType type, size_t numBytes);
  ncclResult_t commCuMemFree(void *ptr);
  ncclResult_t commCuMemAlloc(void **ptr, CUmemGenericAllocationHandle *handlep,
                              CUmemAllocationHandleType type, size_t size);

  std::vector<void *> venusPtrs_;
  std::vector<void *> subPtrs_;
  size_t freeSize_{0};
  void *startPtr_{nullptr};
  CUmemGenericAllocationHandle slabHandle_{};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  size_t totalMemAllocated_{0};
  size_t kMinAlignSize_{16};
};
