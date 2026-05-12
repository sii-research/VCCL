/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "checks.h"
#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "shmutils.h"
#include "p2p.h"
#include "transport.h"
#include <assert.h>
#include "shm.h"
#include "register_inline.h"

NCCL_PARAM(PsmBufferSize, "PSM_BUFFER_SIZE", 64 * 1024 * 1024);
static inline size_t getPsmBufferSize() { return (size_t)ncclParamPsmBufferSize(); }
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
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

struct p2pRegInfo {
  int copyDone;
  int copyStarted;
  int receiverReady;
  void* receiverRegAddr;
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
  struct ncclP2pBuff p2pBuff;
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

extern int64_t ncclParamP2pReadEnable();
extern int64_t ncclParamP2pDirectDisable();

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (!CUDASUCCESS(cudaGetDeviceCount(&ndev)))
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (!CUDASUCCESS(cudaDeviceGetPCIBusId(
            devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i)))
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
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, ret, NULL, &intermediateRank, NULL));
  if (*ret == 0) return ncclSuccess;
  if (intermediateRank != -1) {
    *ret = 0;
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
  if (!CUDASUCCESS(cudaDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2))) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
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
    if (!CUDASUCCESS(cudaIpcGetMemHandle(&ipc, dummy))) {
      INFO(NCCL_INIT|NCCL_P2P,"Legacy IPC not supported");
      *ret = 0;
    }
    NCCLCHECK(ncclCudaFree(dummy));
    legacyIPC = *ret;
    return ncclSuccess;
  }

  if (p2p == 0) {
    INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
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
  INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zu ipcDesc %p", *ptr, size, ipcDesc);

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
      INFO(NCCL_P2P, "UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&cuDesc->data, fd, peer);
      CUCHECK(cuMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      SYSCHECK(close(fd), "close");
    } else {
      CUCHECK(cuMemImportFromShareableHandle(&handle, cuDesc, type));
    }
    CUCHECK(cuMemAddressReserve(&dptr, size, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    CUCHECK(cuMemMap(dptr, size, /* offset */ 0, handle, /* flags */ 0));

    TRACE(NCCL_P2P, "Imported shareable buffer size %zu handle 0x%llx dptr %p", size, handle, (void*)dptr);

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

  INFO(NCCL_P2P, "Imported shareable buffer device %d size %zu ptr %p", comm->cudaDev, size, *devMemPtr);

  return ncclSuccess;
}

// Setting this to non zero causes P2P to use Reads rather than Writes

#define P2P_SAME_PID(MYINFO, PEERINFO) ((MYINFO->hostHash == PEERINFO->hostHash) && (MYINFO->pidHash == PEERINFO->pidHash))

static ncclResult_t p2pGetInfo(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(comm, comm->topo, info1->rank, info2->rank, &p2p, read, intermediateRank, NULL));

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
  useRead = 0;

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
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0 && !ncclParamPassSm()) {
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
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo, sizeof(struct p2pShmProxyInfo)));
  memcpy(&info->desc, &resources->proxyInfo.desc, sizeof(ncclShmIpcDesc_t));
  if (!ncclParamPassSm()) {
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
  useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  recvSize += PSM_BUFFER_SIZE;
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (P2P_SAME_PID(myInfo, peerInfo) && ncclParamP2pDirectDisable() == 0 && useMemcpy == 0 && !ncclParamPassSm()) {
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
        send->conn.buffs[p] = buff;
        buff += PSM_BUFFER_SIZE;
      }
    } else {
      send->conn.buffs[p] = NULL;
    }
  }
  send->conn.stepSize = PSM_BUFFER_SIZE/PSM_STEPS;

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
  recv->conn.stepSize = PSM_BUFFER_SIZE/PSM_STEPS;

  char* buff = (char*)(resources->recvDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (p == NCCL_PROTO_SIMPLE) {
      if (info->read) {
        if (remDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
        /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
        recv->conn.buffs[p] = (char*)(remDevMem+1);
      } else {
        recv->conn.buffs[p] = buff;
        buff += PSM_BUFFER_SIZE;
      }
    } else {
      recv->conn.buffs[p] = NULL;
    }
  }
  if (ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)recv->proxyConn.connection->transportResources;
    proxyInfo->shm = resources->shm;
    proxyInfo->devShm = resources->devShm;
    proxyInfo->desc = resources->desc;
    proxyInfo->recvFifo = recv->conn.buffs[NCCL_PROTO_SIMPLE];
    recv->proxyConn.proxyProgress = psmP2pTransport.recv.proxyProgress;
    NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, NULL, 0, NULL, 0));
  }
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
  if (reqSize == 0) {
    struct p2pShmProxyInfo* proxyInfo;
    size_t shmSize;

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;

    // Create a SHM segment for the peer to attach to
    shmSize = sizeof(struct p2pShm);
    NCCLCHECK(ncclShmAllocateShareableBuffer(shmSize, false, &proxyInfo->desc, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm));
    memset(&proxyInfo->shm->zcAddrExchange, 0, sizeof(proxyInfo->shm->zcAddrExchange));
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
  struct p2pShmProxyInfo* proxyInfo;
  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
  connection->transportResources = proxyInfo;
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
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t psmP2pRecvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<PSM_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t psmP2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  // CE memcpy support
  if (ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      if (proxyInfo->stream) CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<PSM_STEPS; i++) {
        if (proxyInfo->events[i]) CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
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
  if (ncclParamPassSm()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      if (proxyInfo->shm) NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      if (proxyInfo->stream) CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<PSM_STEPS; i++) {
        if (proxyInfo->events[i]) CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      if (ncclCuMemEnable()) {
        psmP2pFreeShareableBuffer(&proxyInfo->p2pBuff.ipcDesc);
        ncclCudaFree(proxyInfo->p2pBuff.directPtr);
      } else {
        ncclCudaFree(proxyInfo->p2pBuff.directPtr);
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
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

#if 0
// CE memcpy support
static ncclResult_t psmP2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile struct ncclConnFifo* connFifo = resources->ceRecvMem->connFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = connFifo[buffSlot].size;
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo+buffSlot*stepSize, resources->ceDevBuff+buffSlot*stepSize, size, cudaMemcpyDeviceToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        cudaError_t res = CUDACLEARERROR(cudaEventQuery(resources->events[buffSlot]));
        if (res != cudaErrorNotReady) CUDACHECK(res);
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          // Notify SHM
          resources->shm->recvMem.tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}
#endif

static inline bool psmP2pProxyReady(struct ncclProxyArgs* args) {
  return args->syncCond == NULL || args->syncCond->proxyReadyEvent.load(std::memory_order_acquire);
}

static inline void psmP2pProxyDone(struct ncclProxyArgs* args) {
  if (args->syncCond) args->syncCond->proxyOpCount.fetch_sub(args->nsubs, std::memory_order_acq_rel);
}

static void psmP2pComputeChunkSize(struct ncclProxySubArgs* sub) {
  size_t chunkBytes = PSM_BUFFER_SIZE;
  if ((size_t)sub->nbytes < PSM_BUFFER_SIZE) {
    size_t mib = sub->nbytes / (1024 * 1024);
    int adjust = mib >= 32 ? 1 :
                 mib >= 16 ? 2 :
                 mib >= 8  ? 4 :
                 mib >= 4  ? 8 :
                 mib >= 2  ? 16 :
                 mib >= 1  ? 32 : 64;
    chunkBytes = PSM_BUFFER_SIZE / adjust;
  }
  sub->chunkSize = chunkBytes / PSM_STEPS;
}

static ncclResult_t psmP2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (!psmP2pProxyReady(args)) return ncclSuccess;
  if (args->protocol != NCCL_PROTO_SIMPLE) {
    WARN("PSM_P2P only supports SIMPLE protocol, got %d", args->protocol);
    return ncclInternalError;
  }

  if (args->reg) {
    if (args->state == ncclProxyOpReady) {
      int readyCount = 0;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
        if (resources->devShm->zcAddrExchange.receiverReady) {
          sub->recvbuff = (uint8_t*)resources->devShm->zcAddrExchange.receiverRegAddr;
          resources->devShm->zcAddrExchange.receiverReady = 0;
          resources->devShm->zcAddrExchange.copyStarted = 0;
          resources->devShm->zcAddrExchange.copyDone = 0;
          sub->done = 0;
          sub->nsteps = 1;
          readyCount++;
        }
      }
      if (readyCount == args->nsubs) {
        args->done = 0;
        args->state = ncclProxyOpProgress;
      } else {
        args->idle = 1;
      }
      return ncclSuccess;
    }
    if (args->state == ncclProxyOpProgress) {
      args->idle = 1;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
        if (sub->sendbuff && sub->recvbuff && !resources->devShm->zcAddrExchange.copyStarted && !resources->devShm->zcAddrExchange.copyDone) {
          CUDACHECK(cudaMemcpyAsync(sub->recvbuff, sub->sendbuff, sub->nbytes, cudaMemcpyDeviceToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[0], resources->stream));
          resources->devShm->zcAddrExchange.copyStarted = 1;
          args->idle = 0;
        }
        if (resources->devShm->zcAddrExchange.copyStarted && !resources->devShm->zcAddrExchange.copyDone) {
          cudaError_t res = CUDACLEARERROR(cudaEventQuery(resources->events[0]));
          if (res == cudaSuccess) {
            resources->devShm->zcAddrExchange.copyDone = 1;
            if (sub->done == 0) {
              sub->done = 1;
              args->done++;
            }
            args->idle = 0;
          } else if (res != cudaErrorNotReady) {
            CUDACHECK(res);
          }
        }
      }
      if (args->done == args->nsubs) {
        args->state = ncclProxyOpNone;
        psmP2pProxyDone(args);
      }
      return ncclSuccess;
    }
    return ncclSuccess;
  }

  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
      psmP2pComputeChunkSize(sub);
      sub->nsteps = DIVUP(sub->nbytes, sub->chunkSize);
      sub->offset = 0;
      resources->shm->sendMem.head = sub->base;
    }
    args->state = ncclProxyOpProgress;
  }

  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
      if (sub->transmitted < sub->done + PSM_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base + sub->transmitted) % PSM_STEPS;
        volatile uint64_t* recvTail = &resources->shm->recvMem.tail;
        if (*recvTail > sub->base + sub->transmitted) {
          size_t size = std::min((size_t)sub->chunkSize, (size_t)(sub->nbytes - sub->offset));
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo + buffSlot * sub->chunkSize,
                                    (char*)sub->sendbuff + sub->offset,
                                    size, cudaMemcpyDeviceToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
          sub->offset += size;
          args->idle = 0;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base + sub->done) % PSM_STEPS;
        cudaError_t res = CUDACLEARERROR(cudaEventQuery(resources->events[buffSlot]));
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          resources->shm->sendMem.head = sub->base + sub->done;
          args->idle = 0;
        } else if (res != cudaErrorNotReady) {
          CUDACHECK(res);
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
      psmP2pProxyDone(args);
    }
  }
  return ncclSuccess;
}

static ncclResult_t psmP2pRecvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (!psmP2pProxyReady(args)) return ncclSuccess;
  if (args->protocol != NCCL_PROTO_SIMPLE) {
    WARN("PSM_P2P only supports SIMPLE protocol, got %d", args->protocol);
    return ncclInternalError;
  }

  if (args->reg) {
    if (args->state == ncclProxyOpReady) {
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
        resources->devShm->zcAddrExchange.receiverRegAddr = sub->recvbuff;
        resources->devShm->zcAddrExchange.receiverRegBytes = sub->nbytes;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        resources->devShm->zcAddrExchange.receiverReady = 1;
        sub->done = 0;
        sub->nsteps = 1;
      }
      args->done = 0;
      args->state = ncclProxyOpProgress;
      return ncclSuccess;
    }
    if (args->state == ncclProxyOpProgress) {
      args->idle = 1;
      for (int s=0; s<args->nsubs; s++) {
        struct ncclProxySubArgs* sub = args->subs+s;
        struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
        if (resources->devShm->zcAddrExchange.copyDone && sub->done == 0) {
          sub->done = 1;
          args->done++;
          resources->devShm->zcAddrExchange.copyDone = 0;
          resources->devShm->zcAddrExchange.copyStarted = 0;
          args->idle = 0;
        }
      }
      if (args->done == args->nsubs) {
        args->state = ncclProxyOpNone;
        psmP2pProxyDone(args);
      }
      return ncclSuccess;
    }
    return ncclSuccess;
  }

  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
      psmP2pComputeChunkSize(sub);
      sub->nsteps = DIVUP(sub->nbytes, sub->chunkSize);
      sub->offset = 0;
      resources->shm->recvMem.tail = sub->base + PSM_STEPS;
    }
    args->state = ncclProxyOpProgress;
  }

  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*)sub->connection->transportResources;
      if (sub->transmitted < sub->done + PSM_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base + sub->transmitted) % PSM_STEPS;
        volatile uint64_t* sendHead = &resources->shm->sendMem.head;
        if (*sendHead > sub->base + sub->transmitted) {
          size_t size = std::min((size_t)sub->chunkSize, (size_t)(sub->nbytes - sub->offset));
          CUDACHECK(cudaMemcpyAsync((char*)sub->recvbuff + sub->offset,
                                    resources->recvFifo + buffSlot * sub->chunkSize,
                                    size, cudaMemcpyDeviceToDevice, resources->stream));
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
          sub->offset += size;
          args->idle = 0;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base + sub->done) % PSM_STEPS;
        cudaError_t res = CUDACLEARERROR(cudaEventQuery(resources->events[buffSlot]));
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          resources->shm->recvMem.tail = sub->base + sub->done + PSM_STEPS;
          args->idle = 0;
        } else if (res != cudaErrorNotReady) {
          CUDACHECK(res);
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
      psmP2pProxyDone(args);
    }
  }
  return ncclSuccess;
}

static ncclResult_t psmP2pProxyRegister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff;
  void* regAddr = NULL;
  ncclResult_t ret = ncclSuccess;
  assert(reqSize % sizeof(struct p2pIpcExpInfo) == 0);
  int numSegments = reqSize/sizeof(struct p2pIpcExpInfo);
  bool* mapped = nullptr;
  bool* imported = nullptr;
  CUmemGenericAllocationHandle* segmentHandles = nullptr;
  size_t totalSize = 0;
  NCCLCHECKGOTO(ncclCalloc(&segmentHandles, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&mapped, numSegments), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&imported, numSegments), ret, fail);
  for (int segment = 0; segment < numSegments; segment++) {
    totalSize += ipcExpInfo[segment].size;
  }
  assert(sizeof(void*) == respSize);

  INFO(NCCL_REG, "Proxy rank %d register reqBuff %p size %zu offset %ld legacyIpcCap %d sameProcess %d, totalSize : %zu", proxyState->tpRank, reqBuff, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess, totalSize);


  // request peer passes all necessary buffer info to import. The proxy thread would register
  // the buffer locally and return register addr back
  if (ipcExpInfo->legacyIpcCap) {
    // legacy import
    CUDACHECKGOTO(cudaIpcOpenMemHandle(&regAddr, ipcExpInfo->ipcDesc.devIpc, cudaIpcMemLazyEnablePeerAccess), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  } else {
    CUCHECKGOTO(cuMemAddressReserve((CUdeviceptr*)&regAddr, totalSize, /* alignment */ 0, /* addr */ 0, /* flags */ 0), ret, fail);
    size_t offset = 0;
    for (int segment = 0; segment < numSegments; segment++) {
      // cuMem import
      if (connection->sameProcess) {
        // if proxy is same process as request peer, we just need to map the handle.
        memcpy(&segmentHandles[segment], &ipcExpInfo[segment].ipcDesc.memHandle, sizeof(CUmemGenericAllocationHandle));
      } else {
        if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
          CUCHECKGOTO(cuMemImportFromShareableHandle(&segmentHandles[segment], (void*)(uintptr_t)ipcExpInfo[segment].impFd, ncclCuMemHandleType), ret, fail);
          SYSCHECKGOTO(close(ipcExpInfo[segment].impFd), "close", ret, fail);
        } else {
          CUCHECKGOTO(cuMemImportFromShareableHandle(&segmentHandles[segment], (void*)&ipcExpInfo[segment].ipcDesc.cuDesc.handle, ncclCuMemHandleType), ret, fail);
        }
      }
      imported[segment] = true;
      CUCHECKGOTO(cuMemMap((CUdeviceptr)regAddr + offset, ipcExpInfo[segment].size, /* offset */ 0, segmentHandles[segment], /* flags */ 0), ret, fail);
      offset += ipcExpInfo[segment].size;
      mapped[segment] = true;
    }
    // Allow access by the local GPU
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = proxyState->cudaDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECKGOTO(cuMemSetAccess((CUdeviceptr)regAddr, totalSize, &accessDesc, 1), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo[0].offset);
  }
  INFO(NCCL_REG, "Proxy rank %d register success regAddr %p size %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, regAddr, ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess);

exit:
  memcpy(respBuff, (void*)&regAddr, sizeof(void*));
  *done = 1;
  free(mapped);
  free(imported);
  free(segmentHandles);
  return ret;
fail:
  if (!ipcExpInfo->legacyIpcCap) {
      size_t offset = 0;
      for (int segment = 0; segment < numSegments; segment++) {
        if (mapped[segment]) {
          CUCHECKIGNORE(cuMemUnmap((CUdeviceptr)regAddr + offset, ipcExpInfo[segment].size));
        }
        if (imported[segment]) {
          CUCHECKIGNORE(cuMemRelease(segmentHandles[segment]));
        }
      }
    if (regAddr) CUCHECKIGNORE(cuMemAddressFree((CUdeviceptr)regAddr, totalSize));
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
      NCCLCHECKGOTO(ncclCuMemFreeAddr((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset), ipcInfo->numSegments), ret, fail);
    } else {
      NCCLCHECKGOTO(ncclCudaFree((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset), ipcInfo->numSegments), ret, fail);
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
  { psmP2pSendSetup, psmP2pSendConnect, psmP2pSendFree, NULL, psmP2pSendProxySetup, psmP2pSendProxyConnect, psmP2pSendProxyFree, psmP2pSendProxyProgress, psmP2pProxyRegister, psmP2pProxyDeregister },
  { psmP2pRecvSetup, psmP2pRecvConnect, psmP2pRecvFree, NULL, psmP2pRecvProxySetup, psmP2pRecvProxyConnect, psmP2pRecvProxyFree, psmP2pRecvProxyProgress, psmP2pProxyRegister, psmP2pProxyDeregister }
};
