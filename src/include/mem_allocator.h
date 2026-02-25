#pragma once

#include <cstdio>
#include <algorithm>
#include <unordered_map>
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

  // Base pointers of all memory pools (each pool is a large slab allocated from CUMEM)
  std::vector<void *> venusPtrs_;
  // Slab size (total bytes) for each memory pool in venusPtrs_
  std::vector<size_t> slabSizes_;
  // Map pool base pointer -> number of live sub-allocations in that pool
  std::unordered_map<void*, size_t> poolLiveCount_;
  // Map sub-allocation pointer -> its owning pool base pointer
  std::unordered_map<void*, void*> subToPool_;
  size_t freeSize_{0};
  void *startPtr_{nullptr};
  CUmemGenericAllocationHandle slabHandle_{};
  size_t granularity_{0};
  CUmemAllocationProp prop_{};
  size_t totalMemAllocated_{0};
  size_t kMinAlignSize_{16};
};
