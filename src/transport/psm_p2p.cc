/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "shmutils.h"
#include <cmath>
#include "p2p.h"
#include "transport.h"
#include <assert.h>
#include <cstddef>
#include "shm.h"
NCCL_PARAM(PsmBufferSize, "PSM_BUFFER_SIZE", 64 * 1024 * 1024);
static inline size_t getPsmBufferSize() {return (size_t)ncclParamPsmBufferSize();}
#define PSM_BUFFER_SIZE getPsmBufferSize()
#define PSM_STEPS 1
enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct ncclP2pRequest {
  size_t size;
  int refcount;
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // Used by CE memcpy
  ncclShmIpcDesc_t desc;
};
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "PSM P2PConnectInfo is too large");

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

struct p2pRegInfo {
  int copyDone;          // Indicates if the copy operation is complete
  int copyStarted;       // Indicates if the copy operation has started
  int receiverReady;     // Indicates if the receiver is ready to receive data
  void* receiverRegAddr; // Address of the receiver's registered memory(shared)
  ssize_t receiverRegBytes;
};
struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
  struct p2pRegInfo zcAddrExchange;
};
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;

  // Intermediate step for sender
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // Receiver buffer
  char* recvFifo;

  // Used by CE memcpy progress only
  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[PSM_STEPS];
  cudaStream_t cpStream;
  // struct ncclP2pBuff p2pBuff; //  TODO
};
static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "PSM P2P Connect info is too large");

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  int sendMemSameProc;
  void* recvMemIpc;
  int recvMemSameProc;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;
};

// cuMem API support
struct p2pCuMemProxyInfo {
  struct ncclP2pBuff p2pBuff;
};

#include <sys/types.h>

// NCCL_PARAM(LegacyCudaRegister, "LEGACY_CUDA_REGISTER", 0);
// NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);
// use origin func to make sure compile success
extern int64_t ncclParamLegacyCudaRegister();
extern int64_t ncclParamP2pReadEnable();
extern int64_t ncclParamP2pDirectDisable();
extern int64_t ncclParamPassSm();
extern int64_t ncclParamMNNVLEnable();

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}
static int useMemcpy = 0;
/* Determine if two peers can communicate through p2p */
ncclResult_t psmP2pCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) {
    if (useMemcpy) *ret = 0;
    return ncclSuccess;
  }

  // Check if NET would work better
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) {
    *ret = 0;
    return ncclSuccess;
  }

  if (info1->hostHash != comm->peerInfo[comm->rank].hostHash ||
      info1->hostHash != info2->hostHash) {
    // If either peer is non-local then we are done.
    return ncclSuccess;
  }

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
#if CUDART_VERSION >= 10010
    // CUDA 10.1 and later can use P2P with invisible devices.
    return ncclSuccess;
#else
    // Peer's CUDA device is not visible in this process : we can't communicate with it.
    *ret = 0;
    return ncclSuccess;
#endif
  }

  // Check that CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != cudaSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"PSM:peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }

  // This will always fail when using NCCL_CUMEM_ENABLE=1
  if (p2p != 0 && !ncclCuMemEnable()) {
    // Cached result of the legacyIPC detection
    static int legacyIPC = -1;
    if (legacyIPC >= 0) {
      *ret = legacyIPC;
      return ncclSuccess;
    }
    // Check that legacy IPC support is available (WSL WAR)
    char *dummy;
    cudaIpcMemHandle_t ipc;
    NCCLCHECK(ncclCudaMalloc(&dummy, CUDA_IPC_MIN));
    if (cudaIpcGetMemHandle(&ipc, dummy) != cudaSuccess) {
      INFO(NCCL_INIT|NCCL_P2P,"PSM:Legacy IPC not supported");
      *ret = 0;
    }
    NCCLCHECK(ncclCudaFree(dummy));
    legacyIPC = *ret;
    return ncclSuccess;
  }

  if (p2p == 0) {
    INFO(NCCL_INIT|NCCL_P2P,"PSM:Could not enable PSM_P2P between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// cuMem API support
static ncclResult_t psmP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    CUmemAllocationHandleType type = ncclCuMemHandleType;

    // cuMem API support
    CUmemGenericAllocationHandle handle;
    NCCLCHECK(ncclCuMemAlloc(ptr, &handle, type, size));
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native cuMem handle for later Export/Import via UDS
      memcpy(&ipcDesc->cuDesc.data, &handle, sizeof(handle));
    } else {
      CUCHECK(cuMemExportToShareableHandle(&ipcDesc->cuDesc, handle, type, 0));
    }
    if (refcount) {
      memcpy(&ipcDesc->memHandle, &handle, sizeof(handle));
      for (int r = 0; r < refcount; ++r) CUCHECK(cuMemRetainAllocationHandle(&handle, *ptr));
    }
#else
    return ncclInternalError;
#endif
  } else {
    // Allocate a CUDA buffer and generate an IPC handle for it
    NCCLCHECK(ncclCudaCalloc((char **)ptr, size));
    cudaError_t res = cudaIpcGetMemHandle(&ipcDesc->devIpc, *ptr);
    if (res != cudaSuccess) {
      WARN("cudaIpcGetMemHandle failed : %s", cudaGetErrorString(res));
      ncclCudaFree(*ptr);
      CUDACHECK(res);
    }
  }
  INFO(NCCL_P2P|NCCL_ALLOC, "PSM:Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc);

  return ncclSuccess;
}

static ncclResult_t psmP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc) {
  return ncclSuccess;
}

static ncclResult_t psmP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr) {
  if (ncclCuMemEnable()) {
#if CUDART_VERSION >= 11030
    // cuMem API support
    CUdeviceptr dptr = 0;
    CUmemAllocationHandleType type = ncclCuMemHandleType;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc *cuDesc = &ipcDesc->cuDesc;
    CUmemAllocationProp prop = {};
    size_t granularity = 0;

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = type;
    prop.location.id = comm->cudaDev;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE(size, granularity);

    // Import and map the remote memory descriptor to the local GPU
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd support
      int fd = -1;
      // Send cuMem handle to remote for conversion to an fd
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, peer, &cuDesc->data, &fd));
      INFO(NCCL_P2P, "PSM:UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&cuDesc->data, fd, peer);
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      SYSCHECK(close(fd), "close");
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&handle, cuDesc, type));
    }
    CUCHECK(cuMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    CUCHECK(cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0));

    TRACE(NCCL_P2P, "PSM:Imported shareable buffer size %zu handle 0x%llx dptr %p", size, handle, (void*)dptr);

    // Allow access by the local GPU
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = comm->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(dptr, size, &accessDesc, 1));
    TRACE(NCCL_P2P, "Set Access for %p size %zu on dev %d", (void*)dptr, size, accessDesc.location.id);

    *devMemPtr = (void *)dptr;
#else
    return ncclInternalError;
#endif
  } else {
    // Legacy CUDA IPC
    CUDACHECK(cudaIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, cudaIpcMemLazyEnablePeerAccess));
  }

  INFO(NCCL_P2P, "PSM:Imported shareable buffer device %d size %zu ptr %p", comm->cudaDev, size, *devMemPtr);

  return ncclSuccess;
}

// Setting this to non zero causes P2P to use Reads rather than Writes
// NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);
// NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

#define P2P_SAME_PID(MYINFO, PEERINFO) ((MYINFO->hostHash == PEERINFO->hostHash) && (MYINFO->pidHash == PEERINFO->pidHash))

static ncclResult_t p2pGetInfo(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, &p2p, read, intermediateRank));

  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {
  if (P2P_SAME_PID(myInfo, peerInfo)) {
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // Same PID different GPUs, enable P2P access
      // Legacy CUDA IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
      if (ncclCuMemEnable()) {
        // for intra-process ranks, we should map memHandle of the peers to increase refcount.
        // Otherwise, if peers abort and free the buffer, the rank can suffer invalid access.
        NCCLCHECK(ncclCuMemAllocAddr(devMem, &p2pBuff->ipcDesc.memHandle, p2pBuff->size));
        CUCHECK(cuMemRelease(p2pBuff->ipcDesc.memHandle));
        *ipcPtr = *devMem;
      } else {
        *devMem = p2pBuff->directPtr;
        *ipcPtr = NULL;
      }
    } else {
      *devMem = p2pBuff->directPtr;
      *ipcPtr = NULL;
    }
  } else {
    // Different PID
    NCCLCHECK(psmP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size, &p2pBuff->ipcDesc, devMem));
    *ipcPtr = *devMem;
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t psmP2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank));
  if (useMemcpy || (ncclParamPassSm() && connIndex == 1)) useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;
  const char* useReadStr = info->read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0 && (!ncclParamPassSm() || connIndex != 1)) {
      resources->type = P2P_DIRECT;
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via PSM_P2P/direct pointer%s",
          channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr);
    } else {
      // cuMem API support
      if (ncclCuMemEnable()) {
        resources->type = P2P_CUMEM;
        const char *MNNVL = comm->MNNVL ? "MNNVL" : "CUMEM";
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via PSM_P2P/%s%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, MNNVL, useReadStr, useMemcpy ? "/CE" : "");;
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via PSM_P2P/IPC%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : "");
      }
    }
    send->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via PSM_P2P/indirect/%d[%d]%s",
        channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, intermediateRank,
        comm->peerInfo[intermediateRank].nvmlDev, useReadStr);
  }

  memset(&req, '\0', sizeof(req));
  req.size = sendSize;
  req.refcount = 0;
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) req.refcount++;
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) req.refcount++;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_PSM_P2P, 1, info->rank, &send->proxyConn));
  if (useMemcpy || (ncclParamPassSm() && connIndex == 1)) {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo, sizeof(struct p2pShmProxyInfo)));
    memcpy(&info->desc, &resources->proxyInfo.desc, sizeof(ncclShmIpcDesc_t));
  } else {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, &send->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  }

  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t psmP2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, myInfo, peerInfo, &useRead, &intermediateRank));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
  if(ncclParamPassSm()) recvSize += PSM_BUFFER_SIZE;
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0 && (!ncclParamPassSm() || connIndex != 1)) {
      resources->type = P2P_DIRECT;
    } else {
      if (ncclCuMemEnable()) {
        // cuMem API support
        resources->type = P2P_CUMEM;
        TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via PSM_P2P/CUMEM",
              channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
      }
    }
    recv->conn.flags |= info->read ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
  }

  memset(&req, '\0', sizeof(req));
  req.size = recvSize;
  req.refcount = 0;
  if (P2P_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].cudaDev != peerInfo->cudaDev)) req.refcount++;
  if (P2P_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].cudaDev != myInfo->cudaDev)) req.refcount++;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_PSM_P2P, 0, info->rank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
  NCCLCHECK(p2pMap(comm, &recv->proxyConn, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->recvDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = P2P_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t psmP2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  struct ncclRecvMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  INFO(NCCL_P2P, "PSM:info->p2pBuff [ send ] = %p", info->p2pBuff.directPtr);
  NCCLCHECK(p2pMap(comm, &send->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));
  char* buff = (char*)(remDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (p == NCCL_PROTO_SIMPLE) {
      if (info->read) {
        /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
        if (resources->sendDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
        send->conn.buffs[p] = (char*)(resources->sendDevMem+1);
      } else {
        /* For PSM */
        send->conn.buffs[p] = buff;
        buff += comm->buffSizes[p];
      }
    } else {
      // PSM dont't support other protocols now
      send->conn.buffs[p] = NULL; 
    }
  }
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/PSM_STEPS;

  if (useMemcpy && !ncclParamPassSm()) {
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail;
    send->conn.connFifo = resources->proxyInfo.ceRecvMem->connFifo;
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head;
    // Send SIMPLE buff to proxy, and replace it by local buffer
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff;
  } else if (ncclParamPassSm()) {
    send->conn.tail = NULL;
    send->conn.head = NULL;
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)send->proxyConn.connection->transportResources;
    send->conn.ptrExchange = (void**)&proxyInfo->devShm->zcAddrExchange.receiverRegAddr;
    send->conn.redOpArgExchange = (uint64_t*)&proxyInfo->devShm->zcAddrExchange.receiverRegBytes;
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0));
  } else {
    send->conn.tail = &remDevMem->tail;
    send->conn.head = &resources->sendDevMem->head;
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
    send->conn.redOpArgExchange = resources->sendDevMem->redOpArgExchange;
  }
  // We must assign the proxyConn's proxyProgress property for proper checking at enqueue-time
  send->proxyConn.proxyProgress = psmP2pTransport.send.proxyProgress;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t psmP2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  struct ncclSendMem* remDevMem = NULL;

  if (useMemcpy || ncclParamPassSm()) {
    // Attach to peer's SHM segment
    NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->shm, (void**)&resources->devShm, &resources->desc));
    recv->conn.tail = &resources->devShm->recvMem.tail;
    recv->conn.head = &resources->devShm->sendMem.head;
    recv->conn.ptrExchange = (void**)&resources->devShm->zcAddrExchange.receiverRegAddr;
    recv->conn.redOpArgExchange = (uint64_t*)&resources->devShm->zcAddrExchange.receiverRegBytes;
  } else {
    NCCLCHECK(p2pMap(comm, &recv->proxyConn, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = P2P_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));

    struct ncclRecvMem* devMem = resources->recvDevMem;
    recv->conn.tail = &devMem->tail;
    recv->conn.head = &remDevMem->head;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
    recv->conn.redOpArgExchange = remDevMem->redOpArgExchange;
  }
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/PSM_STEPS;

  char* buff = (char*)(resources->recvDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (p == NCCL_PROTO_SIMPLE) {
      if (info->read) {
        /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
        if (resources->recvDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
        recv->conn.buffs[p] = (char*)(resources->recvDevMem+1);
      } else {
        recv->conn.buffs[p] = buff;
        buff += comm->buffSizes[p];
      }
    } else {
      // PSM dont't support other protocols now
      recv->conn.buffs[p] = NULL; 
    }
  }
  struct p2pShmProxyInfo *proxyInfo;
  proxyInfo = (struct p2pShmProxyInfo *)recv->proxyConn.connection->transportResources;
  proxyInfo->shm = resources->shm;
  proxyInfo->devShm = resources->devShm;
  proxyInfo->desc = resources->desc;
  proxyInfo->recvFifo = recv->conn.buffs[NCCL_PROTO_SIMPLE];
  recv->proxyConn.proxyProgress = psmP2pTransport.recv.proxyProgress;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, NULL, 0, NULL, 0));
  return ncclSuccess;
}

ncclResult_t psmP2pSendFree(struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc));
        } else {
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
        }
      }
      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc));
        } else {
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
        }
      }
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
    }
    free(resources);
  }
  return ncclSuccess;
}

ncclResult_t psmP2pRecvFree(struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->sendMemIpc));
        } else {
          NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
        }
      }

      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc) {
          NCCLCHECK(ncclCuMemFreeAddr(resources->recvMemIpc));
        } else {
          NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
        }
      }
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
      if (useMemcpy) {
        NCCLCHECK(ncclShmIpcClose(&resources->desc));
      }
    }
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t psmP2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (useMemcpy || ncclParamPassSm()) {
    // CE memcpy support
    struct p2pShmProxyInfo* proxyInfo;
    size_t shmSize;

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;

    // Create a SHM segment for the peer to attach to
    shmSize = sizeof(struct p2pShm);
    NCCLCHECK(ncclShmAllocateShareableBuffer(shmSize, false, &proxyInfo->desc, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm));
    memset(&proxyInfo->devShm->zcAddrExchange, 0, sizeof(proxyInfo->devShm->zcAddrExchange));
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo));
  } else {
    struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
    if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
    int size = req->size;
    if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    NCCLCHECK(psmP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo* proxyInfo;
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
      connection->transportResources = proxyInfo;
    } else {
      connection->transportResources = p2pBuff->directPtr;
    }
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t psmP2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
  if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
  int size = req->size;
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
  NCCLCHECK(psmP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
  p2pBuff->size = size;
  if (ncclCuMemEnable() && !ncclParamPassSm()) {
    // cuMem API support
    struct p2pCuMemProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    connection->transportResources = proxyInfo;
  } else if (ncclCuMemEnable() && ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;
  } else {
    connection->transportResources = p2pBuff->directPtr;
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t psmP2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
  if (reqSize != sizeof(void*)) return ncclInternalError;
  proxyInfo->recvFifo = *((char**)reqBuff);
  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<PSM_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  return ncclSuccess;
}

static ncclResult_t psmP2pRecvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<PSM_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  *done=1;
  return ncclSuccess;
}

static ncclResult_t psmP2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // CE memcpy support
  if (useMemcpy || ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      // NCCLCHECK(ncclCudaHostFree(proxyInfo->ceRecvMem));
      // NCCLCHECK(ncclCudaFree(proxyInfo->ceDevBuff));
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<PSM_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources;
      if (proxyInfo) {
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
        psmP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
        ncclCudaFree(p2pBuff->directPtr);
        free(proxyInfo);
      }
    } else {
      // Do not check return code as CUDA may have already shut down
      ncclCudaFree(connection->transportResources);
    }
  }
  return ncclSuccess;
}

static ncclResult_t psmP2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // CE memcpy support
  if (useMemcpy || ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<PSM_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo *proxyInfo = (struct p2pCuMemProxyInfo *) connection->transportResources;
      if (proxyInfo) {
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
        psmP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
        ncclCudaFree(p2pBuff->directPtr);
        free(proxyInfo);
      }
    } else {
      // Do not check return code as CUDA may have already shut down
      ncclCudaFree(connection->transportResources);
    }
  }
  return ncclSuccess;
}

static void computeChunksize(struct ncclProxySubArgs* sub) {
  size_t dynamic_buffer = PSM_BUFFER_SIZE;
  if (sub->nbytes >= PSM_BUFFER_SIZE) {
    dynamic_buffer = PSM_BUFFER_SIZE;
  } else {
    size_t msize = sub->nbytes / (1024 * 1024);
    int adjustFactor;
    if (msize >= 32) adjustFactor = 1;
    else if (msize >= 16) adjustFactor = 2;
    else if (msize >= 8) adjustFactor = 4;
    else if (msize >= 4) adjustFactor = 8;
    else if (msize >= 2) adjustFactor = 16;
    else if (msize >= 1) adjustFactor = 32;
    else adjustFactor = 64;
    dynamic_buffer = PSM_BUFFER_SIZE / adjustFactor;
  }
  sub->chunkSize = dynamic_buffer / PSM_STEPS;
}

static ncclResult_t psmP2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  //if(!(args->syncCond->proxyReadyEvent.load(std::memory_order_acquire))) return ncclSuccess;
  if (!__atomic_load_n(&(args->syncCond->proxyReadyEvent), __ATOMIC_SEQ_CST)) return ncclSuccess;
  __sync_synchronize();

  if (args->reg) {
    if (args->state == ncclProxyOpReady) {
      int readyCnt = 0;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs + s; 
        volatile struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
        if (resources->devShm->zcAddrExchange.receiverReady) {
          sub->recvbuff = (uint8_t*)resources->devShm->zcAddrExchange.receiverRegAddr;
          resources->devShm->zcAddrExchange.receiverReady = 0;
          resources->devShm->zcAddrExchange.copyStarted = 0;
          resources->devShm->zcAddrExchange.copyDone = 0;
          sub->done = 0;
          sub->nsteps = 1;
          readyCnt++;
        }
      }
      if (readyCnt == args->nsubs) {
        args->done = 0;
        args->state = ncclProxyOpProgress;
      } else {
        args->idle = 1;
      }
      return ncclSuccess;
    }
    else if (args->state == ncclProxyOpProgress) {
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs + s;
        volatile struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
        if (sub->sendbuff && sub->recvbuff && !resources->devShm->zcAddrExchange.copyDone && !resources->devShm->zcAddrExchange.copyStarted) {
          CUDACHECK(cudaMemcpyAsync(sub->recvbuff, sub->sendbuff, sub->nbytes, cudaMemcpyDeviceToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[0], resources->stream));
          resources->devShm->zcAddrExchange.copyStarted = 1;
        }
        
        if (resources->devShm->zcAddrExchange.copyStarted && !resources->devShm->zcAddrExchange.copyDone) {
          cudaError_t res = cudaEventQuery(resources->events[0]);
          if (res == cudaSuccess) {
            resources->devShm->zcAddrExchange.copyDone = 1;
            if (sub->done == 0) {
              sub->done = 1;
              args->done++;
            }
          } else if (res != cudaErrorNotReady){
            CUDACHECK(res);
          }
        }
      }
      if (args->done == args->nsubs) {
        args->state = ncclProxyOpNone;
        //args->syncCond->proxyOpCount.fetch_sub(args->nsubs);
        __sync_synchronize();
        __atomic_fetch_sub(&(args->syncCond->proxyOpCount), args->nsubs, __ATOMIC_SEQ_CST);

      } else {
        args->idle = 1;
      }
      return ncclSuccess;
    }
    return ncclSuccess;
  } else {
    if (args->state == ncclProxyOpReady) {
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
        // Round to next multiple of sliceSteps
        sub->base = ROUNDUP(resources->step, args->chunkSteps);
        sub->posted = sub->transmitted = sub->done = 0;
        // psm specific initilization
        computeChunksize(sub);
        sub->nsteps=(sub->nbytes + sub->chunkSize - 1) / sub->chunkSize;
        resources->shm->sendMem.head = sub->base;
        sub->offset = 0;
      }
      args->state = ncclProxyOpProgress;
    }
    args->idle = 1;
    if (args->state == ncclProxyOpProgress) {
      int p = args->protocol;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
        if (p != NCCL_PROTO_SIMPLE) {
          WARN("PSM P2P transport only supports SIMPLE protocol, got %d", p);
          return ncclInternalError;
        }
        // psm post stage, copy data from userbuffer to recvFifo(receiver shm)
        if (sub->transmitted < sub->done + PSM_STEPS && sub->transmitted < sub->nsteps) {
          int buffSlot = (sub->base + sub->transmitted) % PSM_STEPS;
          volatile u_int64_t* recvTail = &resources->shm->recvMem.tail;
          if (*recvTail > sub->base + sub->transmitted) {
            size_t size = std::min((size_t)sub->chunkSize, sub->nbytes - sub->offset);
            CUDACHECK(cudaMemcpyAsync(
                resources->recvFifo + buffSlot * sub->chunkSize,
                (char*)sub->sendbuff + sub->offset,
                size,
                cudaMemcpyDeviceToDevice, 
                resources->stream));
            CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
            
            sub->transmitted += args->sliceSteps;
            sub->offset += size;
          }
        }
        // psm transmit stage, notify receiver that data is ready to be received
        if (sub->done < sub->transmitted) {
          int buffSlot = (sub->base + sub->done) % PSM_STEPS;
  
          cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
          if (res != cudaErrorNotReady) CUDACHECK(res);
          if (res == cudaSuccess) {
            sub->done += args->sliceSteps;
            // Notify PSM SHM
            volatile uint64_t* sendHead = &resources->shm->sendMem.head;
            *sendHead = sub->base + sub->done;
          }
          if (sub->done == sub->nsteps) {
            resources->step = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
      if (args->done == args->nsubs) {
        args->state = ncclProxyOpNone;
        //args->syncCond->proxyOpCount.fetch_sub(args->nsubs);
        __sync_synchronize();
        __atomic_fetch_sub(&(args->syncCond->proxyOpCount), args->nsubs, __ATOMIC_SEQ_CST);
      }
    }
    return ncclSuccess;
  }
}

static ncclResult_t psmP2pRecvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  //if(!(args->syncCond->proxyReadyEvent.load(std::memory_order_acquire))) return ncclSuccess;
  if (!__atomic_load_n(&(args->syncCond->proxyReadyEvent), __ATOMIC_SEQ_CST)) return ncclSuccess;
  __sync_synchronize();

  if (args->reg) {
    if (args->state == ncclProxyOpReady) {
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs + s;
        volatile struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
          resources->devShm->zcAddrExchange.receiverRegAddr = sub->recvbuff;
          __sync_synchronize();
          resources->devShm->zcAddrExchange.receiverReady = 1;
          sub->done = 0;
          sub->nsteps = 1;
        }
        args->done = 0;
        args->state = ncclProxyOpProgress;
        return ncclSuccess;
      }
      else if (args->state == ncclProxyOpProgress) {
        for (int s=0; s<args->nsubs; s++) {
          struct ncclProxySubArgs* sub = args->subs + s;
          volatile struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)(sub->connection->transportResources);
          
          if (resources->devShm->zcAddrExchange.copyDone == 1 && sub->done == 0) {
            sub->done = 1;
            args->done++;
            resources->devShm->zcAddrExchange.copyDone = 0;
            resources->devShm->zcAddrExchange.copyStarted = 0;
          }
        }
        if (args->done == args->nsubs) {
          args->state = ncclProxyOpNone;
          //args->syncCond->proxyOpCount.fetch_sub(args->nsubs);
          __sync_synchronize();
          __atomic_fetch_sub(&(args->syncCond->proxyOpCount), args->nsubs, __ATOMIC_SEQ_CST);
        } else {
          args->idle = 1;
        }
        return ncclSuccess;
      }
      return ncclSuccess;
  } else {
    if (args->state == ncclProxyOpReady) {
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
        // Round to next multiple of sliceSteps
        sub->base = ROUNDUP(resources->step, args->chunkSteps);
        sub->posted = sub->transmitted = sub->done = 0;
        resources->shm->recvMem.tail = sub->base + PSM_STEPS;
        computeChunksize(sub);
        sub->nsteps=(sub->nbytes + sub->chunkSize - 1) / sub->chunkSize;
        sub->offset = 0;
      }
      args->state = ncclProxyOpProgress;
    }
    args->idle = 1;
    if (args->state == ncclProxyOpProgress) {
      int p = args->protocol;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
  
        if (p != NCCL_PROTO_SIMPLE) {
          WARN("PSM P2P transport only supports SIMPLE protocol, got %d", p);
          return ncclInternalError;
        }
        if (sub->transmitted < sub->done + PSM_STEPS && sub->transmitted < sub->nsteps) {
          int buffSlot = (sub->base + sub->transmitted) % PSM_STEPS;
          volatile uint64_t* sendHead = &resources->shm->sendMem.head;
  
          if ((*sendHead > sub->base + sub->transmitted)) {
            size_t size = std::min((size_t)sub->chunkSize, sub->nbytes - sub->offset);
            CUDACHECK(cudaMemcpyAsync(
                (char*)sub->recvbuff + sub->offset,
                resources->recvFifo + buffSlot * sub->chunkSize,
                size,
                cudaMemcpyDeviceToDevice,
                resources->stream));
            CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
            sub->transmitted += args->sliceSteps;
            sub->offset += size;
          }
        }
        if (sub->done < sub->transmitted) {
          int buffSlot = (sub->base + sub->done) % PSM_STEPS;
          cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
          if (res != cudaErrorNotReady) CUDACHECK(res);
          
          if (res == cudaSuccess) {
            sub->done += args->sliceSteps;
            // Notify sender consume data
            resources->shm->recvMem.tail = sub->base + sub->done + PSM_STEPS;
          }
          if (sub->done == sub->nsteps) {
            resources->step = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
      if (args->done == args->nsubs) {
        args->state = ncclProxyOpNone;
        //args->syncCond->proxyOpCount.fetch_sub(args->nsubs);
        __sync_synchronize();
        __atomic_fetch_sub(&(args->syncCond->proxyOpCount), args->nsubs, __ATOMIC_SEQ_CST);

      }
    }
    return ncclSuccess;
  }
}

static ncclResult_t psmIpcRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc) {
ncclResult_t ret = ncclSuccess;
  struct ncclIpcRegInfo* newInfo = NULL;
  uintptr_t* peerRmtAddrs = NULL;
  int legacyIpcCap = 0;
  size_t baseSize = 0;
  void* baseAddr = NULL;
  bool needUpdate = false;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (isLegacyIpc) *isLegacyIpc = false;
  if (regRecord) {
    // buffer was registered by by users, we need to start to register or reuse it
    int peerLocalRank = -1;
    for (int p = 0; p < nPeers; p++) {
      int peerRank = peerRanks[p];
      peerLocalRank = comm->rankToLocalRank[peerRank];
      if (regRecord->psmIpcInfos[peerLocalRank]) {
        // We already have IPC info for peerLocalRank, no need to register it, we can reuse it
        *regBufFlag = 1;
        if (isLegacyIpc) *isLegacyIpc = regRecord->psmIpcInfos[peerLocalRank]->impInfo.legacyIpcCap;
        INFO(NCCL_REG, "rank %d - IPC reuse buffer %p size %ld (baseAddr %p size %ld) to peer %d regAddr %p", comm->rank, userbuff, buffSize, (void*)regRecord->addr, regRecord->pages * comm->regCache.pageSize, peerRank, regRecord->psmIpcInfos[peerLocalRank]->impInfo.rmtRegAddr);
      } else {
        // Register buffer with peerLocalRank
        struct ncclProxyConnector* proxyConn = NULL;
        struct p2pIpcExpInfo ipcInfo;

        if (baseAddr == NULL) {
          CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail);
          CUCHECKGOTO(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE, (CUdeviceptr)baseAddr), ret, fail);
        }
        if (comm->gproxyConn[peerRank].initialized == false)
          NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PSM_P2P, 1, peerRank, &comm->gproxyConn[peerRank]), ret, fail);
        proxyConn = &comm->gproxyConn[peerRank];

        // Get the mem handle for that buffer. It may have been allocated through cudaMalloc in which case we'll
        // get the CUDA legacy mem handle, or through cuMem*.
        if (ncclCuMemEnable()) {
          CUmemGenericAllocationHandle handle;
          if (CUPFN(cuMemRetainAllocationHandle(&handle, baseAddr)) != CUDA_SUCCESS) {
            // if cuMem* export fails, retry legacy export
            if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
            CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
            ipcInfo.legacyIpcCap = true;
            if (isLegacyIpc) *isLegacyIpc = true;
          } else {
            ipcInfo.legacyIpcCap = false;
            if (isLegacyIpc) *isLegacyIpc = false;
            // cuMem* export to file descriptor or fabric handle
            if (proxyConn->sameProcess) {
              memcpy(&ipcInfo.ipcDesc.memHandle, &handle, sizeof(CUmemGenericAllocationHandle));
            } else {
              if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
                int expFd = -1;
                CUCHECKGOTO(cuMemExportToShareableHandle(&expFd, handle, ncclCuMemHandleType, 0), ret, fail);
                NCCLCHECKGOTO(ncclProxyClientQueryFdBlocking(comm, proxyConn, expFd, &ipcInfo.impFd), ret, fail);
                SYSCHECKGOTO(close(expFd), "close", ret, fail);
              } else {
                // Allow this to silently fail for cases where the user buff cannot be registered
                if (CUPFN(cuMemExportToShareableHandle(&ipcInfo.ipcDesc.cuDesc.handle, handle, ncclCuMemHandleType, 0)) != CUDA_SUCCESS) {
                  CUCHECKGOTO(cuMemRelease(handle), ret, fail);
                  goto fail;
                }
              }
            }
            CUCHECKGOTO(cuMemRelease(handle), ret, fail);
          }
        } else if (legacyIpcCap) {
          // legacy export
          if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
          CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
          ipcInfo.legacyIpcCap = true;
          if (isLegacyIpc) *isLegacyIpc = true;
        } else {
          // nothing works, just return
          goto fail;
        }

        void* rmtRegAddr = NULL;
        ipcInfo.size = baseSize;
        ipcInfo.offset = regRecord->addr - (uintptr_t)baseAddr;
        // Now ipcInfo contains all necessary registration info. Start to register buffer on proxy side
        // and get the remote register address back.
        if (proxyConn) {
          INFO(NCCL_REG, "rank %d - IPC registering buffer %p size %ld (baseAddr %p size %ld) to peer %d", comm->rank, userbuff, buffSize, (void*)regRecord->addr, ipcInfo.size, peerRank);
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgRegister, &ipcInfo, sizeof(p2pIpcExpInfo), &rmtRegAddr, sizeof(void*)), ret, fail);
        }
        if (rmtRegAddr) {
          NCCLCHECKGOTO(ncclCalloc(&newInfo, 1), ret, fail);
          assert(regRecord->psmIpcInfos[peerLocalRank] == NULL);
          regRecord->state |= PSM_P2P_REG_COMPLETE;
          newInfo->peerRank = peerRank;
          newInfo->baseAddr = baseAddr;
          newInfo->impInfo.rmtRegAddr = rmtRegAddr;
          newInfo->impInfo.offset = ipcInfo.offset;
          newInfo->impInfo.legacyIpcCap = ipcInfo.legacyIpcCap;
          newInfo->ipcProxyconn = proxyConn;
          regRecord->psmIpcInfos[peerLocalRank] = newInfo;
          if (regRecord->psmRegIpcAddrs.hostPeerRmtAddrs == NULL) {
            NCCLCHECKGOTO(ncclCalloc(&regRecord->psmRegIpcAddrs.hostPeerRmtAddrs, comm->localRanks), ret, fail);
          }
          regRecord->psmRegIpcAddrs.hostPeerRmtAddrs[peerLocalRank] = (uintptr_t)rmtRegAddr;
          needUpdate = true;
          *regBufFlag = 1;
          INFO(NCCL_REG, "rank %d - IPC registered buffer %p size %ld (baseAddr %p size %ld) to peer %d regAddr %p offsetOut %ld", comm->rank, userbuff, buffSize, (void*)regRecord->addr, ipcInfo.size, peerRank, rmtRegAddr, (uintptr_t)userbuff - regRecord->addr);
        }
      }
    }

    if (*regBufFlag) {
      if (type == NCCL_IPC_COLLECTIVE) {
        // for collective, store registered remote buffers into dev memory for future reference
        if (regRecord->psmRegIpcAddrs.devPeerRmtAddrs == NULL || needUpdate) {
          cudaStream_t hostStream, deviceStream;
          NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false, &hostStream), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
          if (regRecord->psmRegIpcAddrs.devPeerRmtAddrs == NULL)
            NCCLCHECKGOTO(ncclCudaCallocAsync(&regRecord->psmRegIpcAddrs.devPeerRmtAddrs, comm->localRanks, hostStream), ret, fail);
          if (needUpdate)
            NCCLCHECKGOTO(ncclCudaMemcpyAsync(regRecord->psmRegIpcAddrs.devPeerRmtAddrs, regRecord->psmRegIpcAddrs.hostPeerRmtAddrs, comm->localRanks, hostStream), ret, fail);
          NCCLCHECKGOTO(ncclStreamWaitStream(deviceStream, hostStream, comm->sharedRes->scratchEvent), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream, /*concurrent=*/false), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, /*concurrent=*/false), ret, fail);
        }
        peerRmtAddrs = regRecord->psmRegIpcAddrs.devPeerRmtAddrs;
      } else {
        assert(nPeers == 1);
        // p2p always returns remote addr here since remote buffer addr is passed in ncclDevWorkP2p struct
        peerRmtAddrs = (uintptr_t*)regRecord->psmRegIpcAddrs.hostPeerRmtAddrs[peerLocalRank];
      }
      *offsetOut = (uintptr_t)userbuff - regRecord->addr;
      *peerRmtAddrsOut = peerRmtAddrs;
    }
  }
exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (newInfo) free(newInfo);
  INFO(NCCL_REG, "rank %d failed to IPC register userbuff %p buffSize %ld nPeers %d isLegacyIpc %d type %s", comm->rank, userbuff, buffSize, nPeers, isLegacyIpc ? *isLegacyIpc : -1, ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? "POSIX_FD" : "FABRIC");
  goto exit;
}

ncclResult_t psmIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;
  bool isValid = false;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    if (isValid)
      NCCLCHECKGOTO(psmIpcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, NULL), ret, fail);
  }

exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm *comm;
  struct ncclReg *reg;
};

static ncclResult_t cleanupIpc(struct ncclComm* comm, struct ncclCommCallback* cb) {
  struct ncclIpcCleanupCallback* obj = (struct ncclIpcCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  free(obj);
  return ncclSuccess;
}

// ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts) {
static ncclResult_t psmIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  void* baseAddr;
  size_t baseSize;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue = reinterpret_cast<struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>*>(cleanupQueuePtr);
  bool isLegacyIpc = false;
  struct ncclReg *regRecord = NULL;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    CUCHECKGOTO(cuMemGetAddressRange((CUdeviceptr*)&baseAddr, &baseSize, (CUdeviceptr)userbuff), ret, fail);
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, baseAddr, baseSize, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(psmIpcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, &isLegacyIpc), ret, fail);
    if (*regBufFlag) {
      struct ncclIpcCleanupCallback* record;
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
      record->base.fn = cleanupIpc;
      record->comm = comm;
      record->reg = regRecord;
      if (isLegacyIpc) {
        ncclIntruQueueEnqueue(&comm->legacyRegCleanupQueue, (struct ncclCommCallback*)record);
      } else {
        ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
        if (nCleanupQueueElts) *nCleanupQueueElts += 1;
      }
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }

exit:
  // coverity[leaked_storage:FALSE] => normally, addrsRecord is added to the cleanupQueue
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

ncclResult_t psmIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo) {
  NCCLCHECK(ncclProxyCallBlocking(comm, regInfo->ipcProxyconn, ncclProxyMsgDeregister, &regInfo->impInfo, sizeof(struct ncclIpcImpInfo), NULL, 0));
  INFO(NCCL_REG, "rank %d - IPC deregistered buffer %p peer %d ipc remote buffer %p", comm->rank, regInfo->baseAddr, regInfo->peerRank, regInfo->impInfo.rmtRegAddr);
  return ncclSuccess;
}

static ncclResult_t psmP2pProxyRegister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff;
  void* regAddr = NULL;
  ncclResult_t ret = ncclSuccess;
  bool mapped = false;
  bool imported = false;
  CUmemGenericAllocationHandle handle;

  assert(sizeof(struct p2pIpcExpInfo) == reqSize);
  assert(sizeof(void*) == respSize);

  INFO(NCCL_REG, "Proxy rank %d register reqBuff %p size %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, reqBuff, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess);

  // request peer passes all necessary buffer info to import. The proxy thread would register
  // the buffer locally and return register addr back
  if (ipcExpInfo->legacyIpcCap) {
    // legacy import
    CUDACHECKGOTO(cudaIpcOpenMemHandle(&regAddr, ipcExpInfo->ipcDesc.devIpc, cudaIpcMemLazyEnablePeerAccess), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  } else {
    // cuMem import
    if (connection->sameProcess) {
      // if proxy is same process as request peer, we just need to map the handle.
      memcpy(&handle, &ipcExpInfo->ipcDesc.memHandle, sizeof(CUmemGenericAllocationHandle));
    } else {
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        CUCHECKGOTO(cuMemImportFromShareableHandle(&handle, (void*)(uintptr_t)ipcExpInfo->impFd, ncclCuMemHandleType), ret, fail);
        SYSCHECKGOTO(close(ipcExpInfo->impFd), "close", ret, fail);
      } else {
        CUCHECKGOTO(cuMemImportFromShareableHandle(&handle, (void*)&ipcExpInfo->ipcDesc.cuDesc, ncclCuMemHandleType), ret, fail);
      }
    }
    imported = true;
    CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)&regAddr, ipcExpInfo->size, /* alignment */ 0, /* addr */ 0, /* flags */ 0), ret, fail);
    CUCHECKGOTO(cuMemMap((CUdeviceptr)regAddr, ipcExpInfo->size, /* offset */ 0, handle, /* flags */ 0), ret, fail);
    mapped = true;
    // Allow access by the local GPU
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = proxyState->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)regAddr, ipcExpInfo->size, &accessDesc, 1), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  }
  INFO(NCCL_REG, "Proxy rank %d register success regAddr %p size %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, regAddr, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess);

exit:
  memcpy(respBuff, (void*)&regAddr, sizeof(void*));
  *done = 1;
  return ret;
fail:
  if (!ipcExpInfo->legacyIpcCap) {
    if (mapped) CUCHECK(cuMemUnmap((CUdeviceptr)regAddr, ipcExpInfo->size));
    if (regAddr) CUCHECK(cuMemAddressFree((CUdeviceptr)regAddr, ipcExpInfo->size));
    if (imported) CUCHECK(cuMemRelease(handle));
  }
  regAddr = NULL;
  goto exit;
}

static ncclResult_t psmP2pProxyDeregister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcImpInfo* ipcInfo = (struct ncclIpcImpInfo*)reqBuff;
  assert(sizeof(struct ncclIpcImpInfo) == reqSize);

  if (ipcInfo->legacyIpcCap) {
    CUDACHECKGOTO(cudaIpcCloseMemHandle((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
  } else {
    if (connection->sameProcess) {
      NCCLCHECKGOTO(ncclCuMemFreeAddr((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
    } else {
      NCCLCHECKGOTO(ncclCudaFree((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
    }
  }

exit:
  *done = 1;
  return ret;
fail:
  goto exit;
}

struct ncclTransport psmP2pTransport = {
  "PSM_P2P",
  psmP2pCanConnect,
  {
    psmP2pSendSetup, psmP2pSendConnect, psmP2pSendFree, NULL,
    psmP2pSendProxySetup, psmP2pSendProxyConnect, psmP2pSendProxyFree,
    psmP2pSendProxyProgress, psmP2pProxyRegister, psmP2pProxyDeregister
  },
  {
    psmP2pRecvSetup, psmP2pRecvConnect, psmP2pRecvFree, NULL,
    psmP2pRecvProxySetup, psmP2pRecvProxyConnect, psmP2pRecvProxyFree,
    psmP2pRecvProxyProgress, psmP2pProxyRegister, psmP2pProxyDeregister
  }
};