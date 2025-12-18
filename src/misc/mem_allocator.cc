#include "mem_allocator.h"
#include "bitops.h"
#include "checks.h"
#include "cudawrap.h"
#include "nccl.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

MemAllocator::MemAllocator() {
  ncclResult_t result = computeCumemGranualirity();
  if (result != ncclSuccess) {
    std::fprintf(stderr, "Failed to initialize MemAllocator (err=%d)\n",
                 static_cast<int>(result));
    std::abort();
  }
}

ncclResult_t MemAllocator::cuCallocAsync(void **ptr,
                                         CUmemGenericAllocationHandle *handlep,
                                         CUmemAllocationHandleType type,
                                         size_t numBytes) {
  // align allocation size to 16 bytes first, so we make sure startPtr_ is 16
  // bytes aligned
  size_t allocSize = numBytes;
  ALIGN_SIZE(allocSize, kMinAlignSize_);
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  ncclResult_t result = allocateMem(ptr, handlep, type, allocSize);
  if (result != ncclSuccess && result != ncclInProgress) {
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    return result;
  }
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return ncclSuccess;
}

ncclResult_t MemAllocator::allocateMem(void **ptr,
                                       CUmemGenericAllocationHandle *handlep,
                                       CUmemAllocationHandleType type,
                                       size_t numBytes) {
  ncclResult_t result = ncclSuccess;
  if (freeSize_ < numBytes) {
    // No more free space on the last slab, allocate a new one
    // align size to CUMEM_GRANULARITY
    size_t slabSize = numBytes;
    ALIGN_SIZE(slabSize, granularity_);

    void *venusPtr = nullptr;
    NCCLCHECKGOTO(commCuMemAlloc(&venusPtr, &slabHandle_, type, slabSize),
                  result, finish);
    venusPtrs_.push_back(venusPtr);
    slabSizes_.push_back(slabSize);
    poolLiveCount_[venusPtr] = 0;
    *ptr = venusPtr;
    freeSize_ = slabSize - numBytes;
    startPtr_ = (char *)venusPtr + numBytes;
    totalMemAllocated_ += slabSize;
  } else {
    // still there is free space on the last slab, use it and reduce free space
    // count
    *ptr = startPtr_;
    startPtr_ = (char *)startPtr_ + numBytes;
    freeSize_ -= numBytes;
  }
  if (handlep) {
    *handlep = slabHandle_; // copy current slab handle by value
  }
finish:
  // Track this sub-allocation and its owning pool
  void* poolBase = nullptr;
  if (!venusPtrs_.empty()) {
    poolBase = venusPtrs_.back();
  }
  if (poolBase != nullptr && *ptr != nullptr) {
    subToPool_[*ptr] = poolBase;
    poolLiveCount_[poolBase] += 1;
  }
  return result;
}

ncclResult_t MemAllocator::computeCumemGranualirity() {
  CUdevice currentDev;
  int cudaDev;
  CUmemAllocationHandleType type = ncclCuMemHandleType;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop_.requestedHandleTypes = type;
  prop_.location.id = currentDev;
  CUCHECK(cuMemGetAllocationGranularity(&granularity_, &prop_,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return ncclSuccess;
}

ncclResult_t MemAllocator::commCuMemFree(void *ptr) {
  if (ptr == nullptr) {
    return ncclSuccess;
  }
  CUmemGenericAllocationHandle handle;
  CUdeviceptr basePtr;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(&basePtr, &size, (CUdeviceptr)ptr));
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree(basePtr, size));
  return ncclSuccess;
}

ncclResult_t MemAllocator::commCuMemAlloc(void **ptr,
                                          CUmemGenericAllocationHandle *handlep,
                                          CUmemAllocationHandleType type,
                                          size_t size) {
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;

  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, size, granularity, 0, 0));
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
  if (handlep) {
    *handlep = handle;
  }
  return ncclSuccess;
}

bool MemAllocator::ownsPointer(const void* ptr) const {
  if (ptr == nullptr) return false;
  return subToPool_.find(const_cast<void*>(ptr)) != subToPool_.end();
}

ncclResult_t MemAllocator::releaseMem(const void *ptr) {
  if (ptr == nullptr) return ncclSuccess;

  // Lookup the owning pool for this sub-allocation
  auto itSub = subToPool_.find(const_cast<void*>(ptr));
  if (itSub == subToPool_.end()) {
    // Not a tracked sub-allocation; nothing to do
    return ncclSuccess;
  }

  void* poolBase = itSub->second;
  subToPool_.erase(itSub);

  auto itPool = poolLiveCount_.find(poolBase);
  if (itPool == poolLiveCount_.end() || itPool->second == 0) {
    // Inconsistent bookkeeping; do not attempt to free
    return ncclSuccess;
  }

  // Decrease live sub-allocation count for this pool
  itPool->second -= 1;

  // If there are still live sub-allocations, do not free the pool
  if (itPool->second > 0) {
    return ncclSuccess;
  }

  // All sub-allocations under this pool have been released; free the pool
  commCuMemFree(poolBase);

  // Remove this pool from bookkeeping
  poolLiveCount_.erase(itPool);

  // Also erase from venusPtrs_ / slabSizes_ and adjust totalMemAllocated_
  for (size_t i = 0; i < venusPtrs_.size(); ++i) {
    if (venusPtrs_[i] == poolBase) {
      size_t slabSize = slabSizes_[i];
      venusPtrs_.erase(venusPtrs_.begin() + i);
      slabSizes_.erase(slabSizes_.begin() + i);
      if (totalMemAllocated_ >= slabSize) {
        totalMemAllocated_ -= slabSize;
      }
      break;
    }
  }

  return ncclSuccess;
}

MemAllocator::~MemAllocator() {}
