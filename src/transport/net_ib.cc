/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#define NET_IB_CC
#include "timer_log.h"
#include "mlx5/mlx5dvwrap.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

const long long second_to_nanoseconds = 1000000000;

long long get_nanoseconds() {
  long long ns = 0;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  ns += ts.tv_sec;
  ns *= second_to_nanoseconds;
  ns += ts.tv_nsec;
  return ns;
}

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

static int ncclNMergedIbDevs = -1;
#define NCCL_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct alignas(64) ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

static int ncclNIbDevs = -1;
struct alignas(64) ncclIbDev {
  std::mutex mutex;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  char* virtualPciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
};

#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclIbBackupDevs[MAX_IB_DEVS];
static std::mutex ncclIbMutex;
static int ncclIbRelaxedOrderingEnabled = 0;

// With ncclNet_v11_t the NCCL core initializes the network plugin per-communicator
// rather than once for all communicators. However, the internal plugin implementation
// still assumes the plugin is initialized only once across all communicators. The ref
// counter makes sure the plugin internally initializes only once. When per communicator
// context support is added to the plugin the ref counter can be removed.
static int netRefCount;

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", -1);
NCCL_PARAM(IbTc, "IB_TC", -1);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);
NCCL_PARAM(EnableFaultTolerance,"ENABLE_FAULT_TOLERANCE", 0);

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    WARN("communicator encountered a fatal error (detected in %s)\n", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}

// Helper function to convert IB work completion status to string
static const char* ibvWcStatusStr(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS:            return "IBV_WC_SUCCESS";
    case IBV_WC_LOC_LEN_ERR:        return "IBV_WC_LOC_LEN_ERR";
    case IBV_WC_LOC_QP_OP_ERR:      return "IBV_WC_LOC_QP_OP_ERR";
    case IBV_WC_LOC_EEC_OP_ERR:     return "IBV_WC_LOC_EEC_OP_ERR";
    case IBV_WC_LOC_PROT_ERR:       return "IBV_WC_LOC_PROT_ERR";
    case IBV_WC_WR_FLUSH_ERR:       return "IBV_WC_WR_FLUSH_ERR";
    case IBV_WC_MW_BIND_ERR:        return "IBV_WC_MW_BIND_ERR";
    case IBV_WC_BAD_RESP_ERR:       return "IBV_WC_BAD_RESP_ERR";
    case IBV_WC_LOC_ACCESS_ERR:     return "IBV_WC_LOC_ACCESS_ERR";
    case IBV_WC_REM_INV_REQ_ERR:    return "IBV_WC_REM_INV_REQ_ERR";
    case IBV_WC_REM_ACCESS_ERR:     return "IBV_WC_REM_ACCESS_ERR";
    case IBV_WC_REM_OP_ERR:         return "IBV_WC_REM_OP_ERR";
    case IBV_WC_RETRY_EXC_ERR:      return "IBV_WC_RETRY_EXC_ERR";
    case IBV_WC_RNR_RETRY_EXC_ERR:  return "IBV_WC_RNR_RETRY_EXC_ERR";
    case IBV_WC_LOC_RDD_VIOL_ERR:   return "IBV_WC_LOC_RDD_VIOL_ERR";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "IBV_WC_REM_INV_RD_REQ_ERR";
    case IBV_WC_REM_ABORT_ERR:      return "IBV_WC_REM_ABORT_ERR";
    case IBV_WC_INV_EECN_ERR:       return "IBV_WC_INV_EECN_ERR";
    case IBV_WC_INV_EEC_STATE_ERR:  return "IBV_WC_INV_EEC_STATE_ERR";
    case IBV_WC_FATAL_ERR:          return "IBV_WC_FATAL_ERR";
    case IBV_WC_RESP_TIMEOUT_ERR:   return "IBV_WC_RESP_TIMEOUT_ERR";
    case IBV_WC_GENERAL_ERR:        return "IBV_WC_GENERAL_ERR";
    default:                        return "UNKNOWN_STATUS";
  }
}

// Helper function to convert IB work completion opcode to string
static const char* ibvWcOpcodeStr(enum ibv_wc_opcode opcode) {
  switch (opcode) {
    case IBV_WC_SEND:               return "IBV_WC_SEND";
    case IBV_WC_RDMA_WRITE:         return "IBV_WC_RDMA_WRITE";
    case IBV_WC_RDMA_READ:          return "IBV_WC_RDMA_READ";
    case IBV_WC_COMP_SWAP:          return "IBV_WC_COMP_SWAP";
    case IBV_WC_FETCH_ADD:          return "IBV_WC_FETCH_ADD";
    case IBV_WC_BIND_MW:            return "IBV_WC_BIND_MW";
    case IBV_WC_RECV:               return "IBV_WC_RECV";
    case IBV_WC_RECV_RDMA_WITH_IMM: return "IBV_WC_RECV_RDMA_WITH_IMM";
    default:                        return "UNKNOWN_OPCODE";
  }
}
pthread_t ncclIbAsyncThread;
static void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      // Goto Fault Tolerance if enable
      if (!ncclParamEnableFaultTolerance()) ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      // Goto Fault Tolerance if enable
      if (!ncclParamEnableFaultTolerance()) ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      // Goto Fault Tolerance if enable
      if (!ncclParamEnableFaultTolerance()) ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_GID_CHANGE:
      WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->portNum);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->portNum, str, event.event_type);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char* env = ncclGetEnv("NCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char* env = ncclGetEnv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = { 0 };
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr++) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

static ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, *gidIndex, &gid));
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
      return ncclSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate = -1;
    const char* deviceName = wrap_ibv_get_device_name(context->device);
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix)
{
  return (be64toh(subnet_prefix) & 0xffff);
}

static int ncclIbExtractFlid (union ibv_gid *gid)
{
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

static ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  //for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbRoutableFlidIbGidIndex();
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  //for ROCE
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex));
  }

  return ncclSuccess;
}

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

// Returns 0 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // Keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (ncclIbMatchVfPath(p, ncclIbDevs[d].pciPath)) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether RELAXED_ORDERING is enabled and possible
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t r = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == ncclInternalError ? 0 : 1;
}

static bool ncclMlx5dvDmaBufCapable(ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props) {
  if (ncclParamIbMergeNics() == 0 && props->ndevs > 1) {
    INFO(NCCL_NET, "NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
      WARN("NET/IB : Can't make virtual NIC with 0 devices");
      return ncclInvalidUsage;
  }

  if (ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev* mDev = ncclIbMergedDevs + ncclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // Check link layers
  ncclIbDev* dev0 = ncclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link));
      return ncclInvalidUsage;
    }
  }

  *d = ncclNMergedIbDevs++;
  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  std::lock_guard<std::mutex> lock(ncclIbMutex);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props);
  return res;

}

ncclResult_t ncclIbSetNetAttr(void *ctx, ncclNetAttr_t *netAttr) {
  (void)ctx;
  (void)netAttr;
  return ncclSuccess;
}

static ncclProfilerCallback_t ncclProfilerFunction;

static ncclResult_t ncclIbFinalizeDevices(void) {
  netRefCount--;
  return ncclSuccess;
}

static ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  if (netRefCount++) return ret;
  ncclProfilerFunction = profFunction;
  if (ncclParamIbDisable()) return ncclInternalError;
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if(wrap_mlx5dv_symbols() != ncclSuccess) { INFO(NCCL_NET, "NET/IB : Failed to open mlx5dv symbols. Advance features like CX-8 Direct-NIC will be disabled."); }

  if (ncclNIbDevs == -1) {
    std::lock_guard<std::mutex> lock(ncclIbMutex);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "NCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context = NULL;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        char dataDirectDevicePath[PATH_MAX] = "/sys";
        int devCount = /*undefined*/-1, devOffset = 0;
        enum ncclIbProvider ibProvider = wrap_mlx5dv_is_supported(devices[d]) ? IB_PROVIDER_MLX5 : IB_PROVIDER_NONE;

        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }

            // check for mlx5 data direct support only once for a each device
            if (devCount == -1) {
              devCount = 1;
              devOffset = 0;
              if (ncclParamIbDataDirect() > 0 && ibProvider == IB_PROVIDER_MLX5 && ncclMlx5dvDmaBufCapable(context)) {
                int pathLen = strlen(dataDirectDevicePath);
                ncclResult_t res = wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + pathLen, sizeof(dataDirectDevicePath) - pathLen);
                if (res == ncclSuccess) {
                  // data direct devices are exposed twice: with the C2C + PCIe link and with the data direct link
                  devCount = 2;
                  // by default only expose the data direct NIC (devOffset = 1), unless set to 2 by the user
                  devOffset = (ncclParamIbDataDirect() == 2) ? 0 : 1;
                  INFO(NCCL_INIT | NCCL_NET, "NET/IB: Data Direct DMA Interface is detected for device %s", devices[d]->name);
                } else if (res == ncclInvalidArgument) {
                  TRACE(NCCL_NET, "NET/IB: Device %s does not support Data Direct DMA.", devices[d]->name);
                } else {
                  WARN("NET/IB: Error in mlx5dv_get_data_direct_sysfs_path with device %s", devices[d]->name);
                  return res;
                }
              }
            }
            for (int dev = devOffset; dev < devCount; ++dev) {
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
              ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
              ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
              ncclIbDevs[ncclNIbDevs].portNum = port_num;
              ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
              if (portAttr.active_speed_ex) {
                // A non-zero active_speed_ex indicates XDR rate (0x100) or higher
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed_ex) * ncclIbWidth(portAttr.active_width);
              } else {
                ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
              }
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclIbDevs[ncclNIbDevs].pdRefs = 0;
              ncclIbDevs[ncclNIbDevs].pd = NULL;
              if (dev == 0) {
                strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
                NCCLCHECKGOTO(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
              } else {
                snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
                NCCLCHECK(ncclCalloc(&ncclIbDevs[ncclNIbDevs].pciPath, PATH_MAX));
                strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
                ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
              }
              ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
              ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
              ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
              NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

              // Enable ADAPTIVE_ROUTING by default on IB networks
              // But allow it to be overloaded by an env parameter
              ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
              if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

              INFO(NCCL_NET, "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name,
                   ncclIbDevs[ncclNIbDevs].portNum, NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context,
                   ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);

              PTHREADCHECKGOTO(pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs), "pthread_create", ret, fail);
              ncclSetThreadName(ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              PTHREADCHECKGOTO(pthread_detach(ncclIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d

              // Add this plain physical device to the list of virtual devices
              int vDev;
              ncclNetVDeviceProps_t vProps = {0};
              vProps.ndevs = 1;
              vProps.devs[0] = ncclNIbDevs;
              NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps));

              ncclNIbDevs++;
              nPorts++;
            }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
      }

      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; }
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    for (int d = 0; d < ncclNIbDevs; d++) {
        snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
          ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
          ncclIbIfName, ncclSocketToString(&ncclIbIfAddr, addrline));

  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  ncclResult_t ret = ncclSuccess;
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclIbInitDevices(logFunction, profFunction));
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = config->trafficClass;
  *ctx = (void *)netCommConfig;
  return ret;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int ncclIbGdrModuleLoaded = 0; // 1 = true, 0 = false
static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}
ncclResult_t ncclIbGdrSupport() {
  static std::once_flag once;
  std::call_once(once, ibGdrSupportInitOnce);
  if (!ncclIbGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

static __thread int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce(){
  ncclResult_t res;
  int dev_fail = 0;

  // This is a physical device, not a virtual one, so select from ibDevs
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}
// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static std::once_flag onces[MAX_IB_DEVS];
  // init the device only once
  ibDmaSupportInitDev = dev;
  std::call_once(onces[dev], ibDmaBufSupportInitOnce);
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1) return ncclSuccess;
  return ncclSystemError;
}

#define NCCL_NET_IB_MAX_RECVS 8

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  props->regIsGlobal = 1;
  if (ncclIbDmaBufSupport(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

#define NCCL_IB_MAX_QPS 128

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
};

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // FIFO RDMA info
  uint32_t fifoRkey;

  //remote dev info
  union ibv_gid remoteGid;

  // SYNC FIFO RDMA INFO
  uint32_t syncFifoRkey;
};

// Struct containing everything needed to establish connections
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbQpInfo backupQpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbDevInfo backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  char backupDevName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;
  int tc;
  int sl;
  uint64_t syncFifoAddr;
};

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnecting = 6,
  ncclIbCommStateConnected = 7,
  ncclIbCommStatePendingReady = 8,
  ncclIbCommStateSendDevList = 9,
  ncclIbCommStateRecvDevList = 10,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
const char* reqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbNetCommDevBase *backupDevBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  int nreqs;
  long long time;
  int time_out;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int* sizes;
    } recv;
    struct {
      int rank;
    } iput;
  };
  struct timer_log log[NCCL_IB_MAX_DEVS_PER_NIC];
  struct linkStatusTest lTest[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ibv_send_wr retransitionWr;
  int retransitionDevIndex;
  struct ncclIbSendFifo *retransitionElem;
  struct ibv_sge retransitionSge;
};
struct ncclwarn
{
  bool is_warn = false;
  int status;
  int opcode;
  int len;
  int error;
  std::string line;
  std::string type;
  std::string localGidstr;
  std::string localGidstring;
  std::string remoteGidstr;
  std::string remoteGidstring;
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  struct ibv_cq *backupCq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
  alignas(32) struct ncclwarn warn;
};

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint8_t if_backup;
  uint64_t idx;
  char padding[8];
};

// fifo for synchronizing when changing to backup
struct alignas(32) ncclIbSyncFifo {
  uint64_t recvFifoTail;    // get send fifo head, and roll back fifoTail to fifoHead
  uint64_t restartPos;  // Sender sub->transmitted maybe incorrect due to the ack from Receiver is lost
  uint64_t idx;
  int errPortIdx;
};

struct ncclIbRemSyncFifo {
  struct ncclIbSyncFifo elems[MAX_REQUESTS];
  uint64_t syncFifoTail;
  uint64_t addr;
};

struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
  uint8_t srcIp[4];
  uint8_t dscIp[4];
  int channel_id;
  int rank;
  std::string NetworkCardName = "";
  // use for reset qp when using backup qp
  int ib_port;
  ncclIbGidInfo gidInfo;
  uint32_t dest_qp_num;
  struct ncclIbDevInfo info;
  struct ibv_ece ece;
  int ece_supported;
  int tc;
  int sl;
};

struct ncclIbRemSizesFifo {
  int elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t backupRkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC * 2];
  struct ibv_sge sge;
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* fifoMr;
  struct ibv_mr *syncFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
};


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC * 2];
};

struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  ncclNetVDeviceProps_t backupVProps;
  bool isSend;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  struct ncclIbQp backupQps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
  int backupDevIndex;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  int nDataQps;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbDevInfo backupRemDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
  // statistics about the backup comm
  struct ncclIbStats backupStats;
};

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with fifo and ibv structs as they have alignment restrictions
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbSendCommDev backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ncclIbRemSizesFifo remSizesFifo;
  uint64_t fifoHead;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;
  int backupAr;
  uint8_t func;
  unsigned long long ncclFuncTimes;
  int peerRank;
  int rank;
  uint64_t groupHash;
  struct ncclIbSyncFifo syncFifo[MAX_REQUESTS];
  uint64_t syncFifoHead; // use for syncFifo
  int sendCcCnt;         // use for refresh devstate, when sendCcCnt % 600 == 0, we can refresh devstate
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure fifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  struct ibv_mr* fifoMr;
  struct ibv_sge fifoSge;
  struct ibv_mr* sizesFifoMr;
  struct ibv_mr *syncFifoMr;
  struct ibv_sge syncFifoSge;
};

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRecvCommDev backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
  int backupFlushEnabled;
  struct ncclIbRemSyncFifo remSyncFifo;
  int recvCcCnt; // use for refresh devstate, when recvCcCnt % 600 == 0, we can refresh devstate
};
static_assert((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbRecvComm fifo must be 32-byte aligned");

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

static void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, struct ncclIbNetCommDevBase* base, bool if_backup) {
  req->events[devIndex]++;
  if (!if_backup) {
    req->devBases[devIndex] = base;
  }
  else {
    req->backupDevBases[devIndex] = base;
  }
}
ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context, bool if_backup) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  if (!if_backup) {
    NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, 2 * MAX_REQUESTS * ncclParamIbQpsPerConn(), cq_context, NULL, 0));
  }
  else {
    NCCLCHECK(wrap_ibv_create_cq(&base->backupCq, ibDev->context, 2 * MAX_REQUESTS * ncclParamIbQpsPerConn(), cq_context, NULL, 0));
  }

  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base, bool if_backup) {
  if (!if_backup) {
    NCCLCHECK(wrap_ibv_destroy_cq(base->cq));
  }
  else {
    NCCLCHECK(wrap_ibv_destroy_cq(base->backupCq));
  }

  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* base, int access_flags, void* qp_context, struct ncclIbQp* qp, bool if_backup) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = qp_context;
  if (!if_backup) {
    qpInitAttr.send_cq = base->cq;
    qpInitAttr.recv_cq = base->cq;
  }
  else {
    qpInitAttr.send_cq = base->backupCq;
    qpInitAttr.recv_cq = base->backupCq;
  }
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS + 1; // +1 for retransition wr
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  TRACE(NCCL_NET, "NET/IB : ncclIbCreateQp port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qpn=%u pkey=%u pd=%p",
    ib_port, base->ibDevN, ncclIbDevs[base->ibDevN].devName, ncclNIbDevs, ncclNMergedIbDevs, qp->qp->qp_num, qpAttr.pkey_index, base->pd);
  qp->ib_port = ib_port;
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* sGidInfo, uint32_t dest_qp_num, struct ncclIbDevInfo* info, bool fifoTc, int tc, int sl) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = fifoTc && ncclParamIbFifoTc() != -1 ? ncclParamIbFifoTc() : tc;
  } else {
    //pick lid if subnet prefixs are same, FLID if they are not
    if (ncclIbExtractLocalSubnetPrefix(sGidInfo->localGid.global.subnet_prefix) ==
		    ncclIbExtractLocalSubnetPrefix(info->gid.global.subnet_prefix)) {
        qpAttr.ah_attr.is_global = 0;
        qpAttr.ah_attr.dlid = info->lid;
    } else {
	      uint16_t flid = ncclIbExtractFlid(&info->gid);
        if (flid == 0) {
          WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as fallback");
          qpAttr.ah_attr.dlid = info->lid;
	      } else {
          qpAttr.ah_attr.dlid = ncclIbExtractFlid(&info->gid);
	      }
        qpAttr.ah_attr.is_global = 1;
        qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
        qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
        qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
	      qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  qpAttr.ah_attr.sl = sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  TRACE(NCCL_NET, "NET/IB : ncclIbRtrQp qpn=%u mtu=%d dst=%u ll=%u port=%u sl: %d tc: %d", qp->qp_num, info->mtu, dest_qp_num, info->link_layer, info->ib_port, qpAttr.ah_attr.sl, qpAttr.ah_attr.grh.traffic_class);
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;
  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  if (stage->state == ncclIbCommStateConnect)      goto ib_connect_check;
  if (stage->state == ncclIbCommStateSendDevList)  goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecvDevList)  goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSend)         goto ib_send;
  if (stage->state == ncclIbCommStateConnecting)   goto ib_connect;
  if (stage->state == ncclIbCommStateConnected)    goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&comm->base.stats), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  comm->base.isSend = true;
  stage->state = ncclIbCommStateSendDevList;
  stage->offset = 0;
  struct ncclIbConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  ncclNetCommConfig_t* config;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  // backup dev
  int backupDev;
  backupDev = dev ^ 1;
  struct ncclIbMergedDev *backupMergedDev;
  backupMergedDev = ncclIbMergedDevs + backupDev;
  comm->base.backupVProps = backupMergedDev->vProps;

  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  comm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  if (ncclParamEnableFaultTolerance()) comm->backupAr = 1; // Set to 1 for logic
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats, false), ret, fail);
    comm->ar = comm->ar && ncclIbDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled

    // backup dev
    if (ncclParamEnableFaultTolerance()) {
      int backupIbDevN = comm->base.backupVProps.devs[i];
      NCCLCHECKGOTO(ncclIbInitCommDevBase(backupIbDevN, &comm->backupDevs[i].base, &comm->base.backupStats, true), ret, fail);
      comm->backupAr = comm->backupAr && ncclIbDevs[backupIbDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
    }
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  // backupDev use same port index with devIndex
  int backupDevIndex;
  backupDevIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    ncclIbSendCommDev* commDev = comm->devs + devIndex;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &commDev->base, IBV_ACCESS_REMOTE_WRITE, &comm->base.stats, comm->base.qps + q, false), ret, fail);
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn      = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // backup QP
    if (ncclParamEnableFaultTolerance()) {
      ncclIbSendCommDev *backupCommDev = comm->backupDevs + backupDevIndex;
      ncclIbDev *backupIbDev = ncclIbDevs + backupCommDev->base.ibDevN;
      NCCLCHECK(ncclIbCreateQp(backupIbDev->portNum, &backupCommDev->base, IBV_ACCESS_REMOTE_WRITE, &comm->base.backupStats, comm->base.backupQps + q, true));
      comm->base.backupQps[q].devIndex = backupDevIndex;
      meta.backupQpInfo[q].qpn = comm->base.backupQps[q].qp->qp_num;
      meta.backupQpInfo[q].devIndex = comm->base.backupQps[q].devIndex;
    }

    if (ncclParamIbEceEnable()) {
      // Query ece capabilities (enhanced connection establishment)
      NCCLCHECKGOTO(wrap_ibv_query_ece(comm->base.qps[q].qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
      if (ncclParamEnableFaultTolerance()) NCCLCHECKGOTO(wrap_ibv_query_ece(comm->base.backupQps[q].qp, &meta.backupQpInfo[q].ece, &meta.backupQpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }
    devIndex = (devIndex + 1) % comm->base.vProps.ndevs;
    if (ncclParamEnableFaultTolerance()) backupDevIndex = (backupDevIndex + 1) % comm->base.backupVProps.ndevs;
  }

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;

    ncclIbSendCommDev *backupCommDev = NULL;
    ncclIbDev *backupIbDev = NULL;
    if (ncclParamEnableFaultTolerance()) {
      backupCommDev = comm->backupDevs + i;
      backupIbDev = ncclIbDevs + backupCommDev->base.ibDevN;
    }

    // Write to the metadata struct via this pointer
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port       = ibDev->portNum;
    devInfo->mtu           = ibDev->portAttr.active_mtu;
    devInfo->lid           = ibDev->portAttr.lid;

    // Write backup info to the metadata struct via this pointer
    ncclIbDevInfo *backupDevInfo = NULL;
    if (ncclParamEnableFaultTolerance()) {
      backupDevInfo = meta.backupDevs + i;
      backupDevInfo->ib_port = backupIbDev->portNum;
      backupDevInfo->mtu = backupIbDev->portAttr.active_mtu;
      backupDevInfo->lid = backupIbDev->portAttr.lid;
    } 

    // Prepare GIN Put Signal scratchpad (for RDMA Atomic result)
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr, commDev->base.pd, &comm->putSignalScratchpad, sizeof(comm->putSignalScratchpad), IBV_ACCESS_LOCAL_WRITE), ret, fail);

    // Prepare my fifo
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->fifoMr, commDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // Prepare backup fifo
    if (ncclParamEnableFaultTolerance()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupCommDev->fifoMr, backupCommDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo) * MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      backupDevInfo->fifoRkey = backupCommDev->fifoMr->rkey;
    }

    // Prepare syncFifo
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->syncFifoMr, commDev->base.pd, comm->syncFifo, sizeof(struct ncclIbSyncFifo) * MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->syncFifoRkey = commDev->syncFifoMr->rkey;

    // Prepare backup syncFifo
    if (ncclParamEnableFaultTolerance()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupCommDev->syncFifoMr, backupCommDev->base.pd, comm->syncFifo, sizeof(struct ncclIbSyncFifo) * MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      backupDevInfo->syncFifoRkey = backupCommDev->syncFifoMr->rkey;
    }

    // Pack local GID info
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &commDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, commDev->base.gidInfo.localGidIndex, &commDev->base.gidInfo.localGid), ret, fail);
    devInfo->gid.global.subnet_prefix = commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id = commDev->base.gidInfo.localGid.global.interface_id;

    // info logging
    for (int q = 0; q < comm->base.nqps; q++) {
      // Print just the QPs for this dev
      if (comm->base.qps[q].devIndex == i) {
        if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d subnet-prefix %lu  FLID %d fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
               dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
               (uint64_t)devInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&devInfo->gid), devInfo->fifoRkey, commDev->fifoMr->lkey);
        } else { // RoCE
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu,
               (int64_t)commDev->base.gidInfo.localGidIndex,
               (uint64_t)devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, devInfo->fifoRkey, commDev->fifoMr->lkey);
        }
        // Log ECE info
        if (meta.qpInfo[q].ece_supported) {
          INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               meta.qpInfo[q].ece_supported, meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask);
        }
      }
    }

    // backup Pack local GID info
    if (ncclParamEnableFaultTolerance()) {
      backupDevInfo->link_layer = backupCommDev->base.gidInfo.link_layer = backupIbDev->portAttr.link_layer;
      NCCLCHECKGOTO(ncclIbGetGidIndex(backupIbDev->context, backupIbDev->portNum, &backupIbDev->portAttr, &backupCommDev->base.gidInfo.localGidIndex), ret, fail);
      NCCLCHECKGOTO(wrap_ibv_query_gid(backupIbDev->context, backupIbDev->portNum, backupCommDev->base.gidInfo.localGidIndex, &backupCommDev->base.gidInfo.localGid), ret, fail);
      backupDevInfo->gid.global.subnet_prefix = backupCommDev->base.gidInfo.localGid.global.subnet_prefix;
      backupDevInfo->gid.global.interface_id = backupCommDev->base.gidInfo.localGid.global.interface_id;

      // backup info logging
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.backupQps[q].devIndex == i) {
          if (backupDevInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
            INFO(NCCL_NET, "NET/IB: %s %d backupIbDev %d Port %d qpn %d mtu %d LID %d subnet-prefix %lu  FLID %d fifoRkey=0x%x fifoLkey=0x%x",
                comm->base.backupVProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
                backupDev, backupCommDev->base.ibDevN, backupIbDev->portNum, meta.backupQpInfo[q].qpn, backupDevInfo->mtu, backupDevInfo->lid,
                (uint64_t)backupDevInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&backupDevInfo->gid), backupDevInfo->fifoRkey, backupCommDev->fifoMr->lkey);
          }
          else { // RoCE
            INFO(NCCL_NET, "NET/IB: %s %d backupIbDev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
                comm->base.backupVProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", backupDev,
                backupCommDev->base.ibDevN, backupIbDev->portNum, meta.backupQpInfo[q].qpn, backupDevInfo->mtu,
                (int64_t)backupCommDev->base.gidInfo.localGidIndex,
                (uint64_t)backupDevInfo->gid.global.subnet_prefix, backupDevInfo->gid.global.interface_id, backupDevInfo->fifoRkey, backupCommDev->fifoMr->lkey);
          }
          // Log ECE info
          if (meta.backupQpInfo[q].ece_supported){
            INFO(NCCL_NET, "NET/IB: backupIbDev %d Port %d qpn %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
                backupCommDev->base.ibDevN, backupIbDev->portNum, meta.backupQpInfo[q].qpn,
                meta.backupQpInfo[q].ece_supported, meta.backupQpInfo[q].ece.vendor_id, meta.backupQpInfo[q].ece.options, meta.backupQpInfo[q].ece.comp_mask);
          }
        }
      }
    }

    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }
  config = (ncclNetCommConfig_t*)ctx;
  meta.fifoAddr = (uint64_t)comm->fifo;
  meta.syncFifoAddr = (uint64_t)comm->syncFifo;
  meta.sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_SL_DEFAULT;
  meta.tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);
  if (ncclParamEnableFaultTolerance()) strncpy(meta.backupDevName, backupMergedDev->devName, MAX_MERGED_DEV_NAME);

  for (int q = 0; q < comm->base.nqps; q++) {
    *(u_int *)comm->base.qps[q].srcIp = *(u_int *)(&comm->devs[comm->base.qps[q].devIndex].base.gidInfo.localGid.raw[12]);
    comm->base.qps[q].NetworkCardName = meta.devName;

    if (ncclParamEnableFaultTolerance()) {
      *(u_int *)comm->base.backupQps[q].srcIp = *(u_int *)(&comm->backupDevs[comm->base.backupQps[q].devIndex].base.gidInfo.localGid.raw[12]);
      if (meta.backupDevName[0] != '\0')
        comm->base.backupQps[q].NetworkCardName = meta.backupDevName;
    }
  }

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct ncclIbConnectionMetadata remMeta;
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;

  // ensure that the remote devices have the same link layer than the local devices used in the connection.
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = ncclIbDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }

    if (ncclParamEnableFaultTolerance()) {
      int backupIbDev0 = comm->backupDevs[0].base.ibDevN;
      link_layer = ncclIbDevs[backupIbDev0].portAttr.link_layer;
      for (int i = 0; i < remMeta.ndevs; i++) {
        if (remMeta.backupDevs[i].link_layer != link_layer) {
          WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
              NCCL_IB_LLSTR(remMeta.backupDevs[i].link_layer), backupIbDev0, ncclIbDevs[backupIbDev0].devName, ncclIbDevs[backupIbDev0].portNum, NCCL_IB_LLSTR(link_layer));
          return ncclInternalError;
        }
      }
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;

    if (ncclParamEnableFaultTolerance()) {
      comm->base.backupRemDevs[i] = remMeta.backupDevs[i];
      comm->base.backupRemDevs[i].remoteGid.global.interface_id = comm->base.backupRemDevs[i].gid.global.interface_id;
      comm->base.backupRemDevs[i].remoteGid.global.subnet_prefix = comm->base.backupRemDevs[i].gid.global.subnet_prefix;
    }

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;

    if (ncclParamEnableFaultTolerance()) {
      comm->remSizesFifo.backupRkeys[i] = remMeta.backupDevs[i].fifoRkey;
    }
  }

  for(int q = 0; q < comm->base.nqps; q++) {
    struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
    *(u_int *)comm->base.qps[q].dscIp = *(u_int*)(&comm->base.remDevs[remQpInfo->devIndex].remoteGid.raw[12]);

    if (ncclParamEnableFaultTolerance()) {
      struct ncclIbQpInfo* backupRemQpInfo = remMeta.backupQpInfo + q;
      *(u_int *)comm->base.backupQps[q].dscIp = *(u_int*)(&comm->base.backupRemDevs[backupRemQpInfo->devIndex].remoteGid.raw[12]);
    }
  }

  for (int i=0; i < comm->base.vProps.ndevs; i++) {
    NCCLCHECKGOTO(wrap_ibv_reg_mr(comm->remSizesFifo.mrs+i, comm->devs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    if (ncclParamEnableFaultTolerance()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(comm->remSizesFifo.mrs + i + NCCL_IB_MAX_DEVS_PER_NIC, comm->backupDevs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int) * MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
    }
  }
  comm->base.nRemDevs = remMeta.ndevs;

  for (int q = 0; q < comm->base.nqps; q++) {
    struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
    struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;

    struct ncclIbQpInfo *backupRemQpInfo = NULL;
    struct ncclIbDevInfo *backupRemDevInfo = NULL;
    if (ncclParamEnableFaultTolerance()) {
      backupRemQpInfo = remMeta.backupQpInfo + q;
      backupRemDevInfo = remMeta.backupDevs + backupRemQpInfo->devIndex;
    }

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    ncclIbSendCommDev* commDev = comm->devs + devIndex;

    struct ibv_qp* qp = comm->base.qps[q].qp;
    if (remQpInfo->ece_supported) {
      struct ncclIbQp* nqp = comm->base.qps + q;
      int ibDevN = comm->devs[nqp->devIndex].base.ibDevN;
      struct ncclIbDev* ibDev = ncclIbDevs + ibDevN;
      INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        ibDevN, ibDev->portNum, qp->qp_num, remMeta.qpInfo[q].ece_supported, remMeta.qpInfo[q].ece.vendor_id, remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remQpInfo->ece, &remQpInfo->ece_supported), ret, fail);
    }

    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    NCCLCHECKGOTO(ncclIbRtrQp(qp, &commDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);

    memcpy(&comm->base.qps[q].gidInfo, &commDev->base.gidInfo, sizeof(struct ncclIbGidInfo));
    comm->base.qps[q].dest_qp_num = remQpInfo->qpn;
    memcpy(&comm->base.qps[q].info, remDevInfo, sizeof(struct ncclIbDevInfo));
    comm->base.qps[q].ece_supported = remQpInfo->ece_supported;
    comm->base.qps[q].ece = remQpInfo->ece;
    comm->base.qps[q].tc = remMeta.tc;
    comm->base.qps[q].sl = remMeta.sl;

    // Assign per-QP backup remDev
    if (ncclParamEnableFaultTolerance()) {
      comm->base.backupQps[q].remDevIdx = backupRemQpInfo->devIndex;
      devIndex = comm->base.backupQps[q].devIndex;
      ncclIbSendCommDev *backupCommDev = comm->backupDevs + devIndex;

      qp = comm->base.backupQps[q].qp;
      if (backupRemQpInfo->ece_supported) {
        struct ncclIbQp *nqp = comm->base.backupQps + q;
        int backupIbDevN = comm->backupDevs[nqp->devIndex].base.ibDevN;
        struct ncclIbDev *backupIbDev = ncclIbDevs + backupIbDevN;
        INFO(NCCL_NET, "NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
          backupIbDevN, backupIbDev->portNum, qp->qp_num, remMeta.backupQpInfo[q].ece_supported, remMeta.backupQpInfo[q].ece.vendor_id, remMeta.backupQpInfo[q].ece.options, remMeta.backupQpInfo[q].ece.comp_mask);
        NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &backupRemQpInfo->ece, &backupRemQpInfo->ece_supported), ret, fail);
      }

      ncclIbDev *backupIbDev = ncclIbDevs + backupCommDev->base.ibDevN;
        backupRemDevInfo->mtu = std::min(backupRemDevInfo->mtu, backupIbDev->portAttr.active_mtu);
      NCCLCHECKGOTO(ncclIbRtrQp(qp, &backupCommDev->base.gidInfo, backupRemQpInfo->qpn, backupRemDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);
    }
  }

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, comm->base.nRemDevs);

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  return ret;
fail:
  free(comm);
  goto exit;
}

NCCL_PARAM(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

ncclResult_t ncclIbCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
  ncclNetVDeviceProps_t  outVProps = {0};
  ncclNetVDeviceProps_t* minVProps = vProps2;
  ncclNetVDeviceProps_t* maxVProps = vProps1;
  if (vProps2->ndevs > vProps1->ndevs) {
    minVProps = vProps1;
    maxVProps = vProps2;
  }

  // Find the intersection of devices
  for (int i = 0; i < minVProps->ndevs; i++) {
    int dev = minVProps->devs[i];
    for (int j = 0; j < maxVProps->ndevs; j++) {
      // Found
      if (maxVProps->devs[j] == dev) {
        outVProps.devs[outVProps.ndevs++] = dev;
      }
    }
  }

  // In the case that at least one side has a fused NIC but there are no matching physical NICs, we should check if the user wants this
  if (ncclParamIbWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
    char local[128];
    int cursor = 1;
    snprintf(local, sizeof(local), "%d", vProps1->devs[0]);
    for (int i = 1; i < vProps1->ndevs; i++) {
      snprintf(local+cursor, sizeof(local)-cursor, ",%d", vProps1->devs[i]);
      cursor += 2;
    }
    char remote[128];
    snprintf(remote, sizeof(remote), "%d", vProps2->devs[0]);
    cursor = 1;
    for (int i = 1; i < vProps2->ndevs; i++) {
      snprintf(remote+cursor, sizeof(remote)-cursor, ",%d", vProps2->devs[i]);
      cursor += 2;
    }
    INFO(NCCL_NET, "NET/IB : There are mismatched physical devices between local (%s) and remote (%s). To disable this warning, set NCCL_IB_WARN_RAIL_LOCAL=0", local, remote);
  }

  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *recvComm = NULL;

  if (stage->state == ncclIbCommStateAccept)   goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  // Alloc stage->buffer here to be used for all following steps
  struct ncclIbConnectionMetadata remMeta;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;
  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  // Reduce the physical device list and store in the connection base
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + lComm->dev;
  struct ncclIbMergedDev *backupMergedDev;
  backupMergedDev = ncclIbMergedDevs + (lComm->dev ^ 1);
  NCCLCHECK(ncclIbCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  rComm->base.backupVProps = backupMergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  rComm->base.isSend = false;
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * rComm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  rComm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  stage->offset = 0;
  stage->state = ncclIbCommStateSendDevList;

ib_send_dev_list:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset), ret, fail);
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

  // IB setup
  // Pre-declare variables because of goto
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;
  struct ncclIbDevInfo* remDevInfo;
  struct ncclIbQp* qp;

  struct ncclIbDev *backupIbDev;
  int backupIbDevN;
  struct ncclIbRecvCommDev *backupRCommDev;
  struct ncclIbDevInfo *backupRemDevInfo;
  struct ncclIbQp *backupQp;

  // To prevent compile warning when disable Fault Tolerance
  backupRCommDev = NULL;
  backupRemDevInfo = NULL;
  backupQp = NULL;
  backupIbDev = NULL;

  mergedDev = ncclIbMergedDevs + lComm->dev;
  backupMergedDev = ncclIbMergedDevs + (lComm->dev ^ 1);
  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
      mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, rComm->base.nRemDevs);
  }

  // Metadata to send back to requestor (sender)
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats, false), ret, fail);
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &rCommDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, rCommDev->base.gidInfo.localGidIndex, &rCommDev->base.gidInfo.localGid), ret, fail);

    // backup
    if (ncclParamEnableFaultTolerance()) {
      backupRCommDev = rComm->backupDevs + i;
      backupIbDevN = rComm->base.backupVProps.devs[i];
      NCCLCHECKGOTO(ncclIbInitCommDevBase(backupIbDevN, &backupRCommDev->base, &rComm->base.backupStats, true), ret, fail);
      backupIbDev = ncclIbDevs + backupIbDevN;
      NCCLCHECKGOTO(ncclIbGetGidIndex(backupIbDev->context, backupIbDev->portNum, &backupIbDev->portAttr, &backupRCommDev->base.gidInfo.localGidIndex), ret, fail);
      NCCLCHECKGOTO(wrap_ibv_query_gid(backupIbDev->context, backupIbDev->portNum, backupRCommDev->base.gidInfo.localGidIndex, &backupRCommDev->base.gidInfo.localGid), ret, fail);
    }

    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;

    // back up
    if (ncclParamEnableFaultTolerance()) {
      rComm->base.backupRemDevs[i] = remMeta.backupDevs[i];
      rComm->base.backupRemDevs[i].remoteGid.global.interface_id = rComm->base.backupRemDevs[i].gid.global.interface_id;
      rComm->base.backupRemDevs[i].remoteGid.global.subnet_prefix = rComm->base.backupRemDevs[i].gid.global.subnet_prefix;
    }

    if (remMeta.devs[i].link_layer != link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIndex;
  int backupRemDevIndex;
  int devIndex;
  int backupDevIndex;
  devIndex = 0;
  backupDevIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIndex = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIndex;
    qp = rComm->base.qps+q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIndex;

    if (ncclParamEnableFaultTolerance()) {
      backupRemDevIndex = remMeta.backupQpInfo[q].devIndex;
      backupRemDevInfo = remMeta.backupDevs + backupRemDevIndex;
      backupQp = rComm->base.backupQps + q;
      backupRCommDev = rComm->backupDevs + backupDevIndex;
      backupQp->remDevIdx = backupRemDevIndex;
    }

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC, &rComm->base.stats, qp, false), ret, fail);
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.vProps.ndevs;

    // back up local ibDevN
    if (ncclParamEnableFaultTolerance()) {
      backupIbDevN = rComm->backupDevs[backupDevIndex].base.ibDevN;
      backupIbDev = ncclIbDevs + backupIbDevN;
      NCCLCHECKGOTO(ncclIbCreateQp(backupIbDev->portNum, &backupRCommDev->base, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC, &rComm->base.backupStats, backupQp, true), ret, fail);
      backupQp->devIndex = backupDevIndex;
      backupDevIndex = (backupDevIndex + 1) % rComm->base.backupVProps.ndevs;
    }

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].ece_supported) {
      // Coverity suspects a copy-paste error below due to the use of remMeta in one argument and meta in another.
      // However, this has been confirmed to be intentional.
      // coverity[copy_paste_error]
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp->qp, &remMeta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }

    NCCLCHECKGOTO(ncclIbRtrQp(qp->qp, &rCommDev->base.gidInfo, remMeta.qpInfo[q].qpn, remDevInfo, true, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp->qp), ret, fail);

    // Query the reduced ece for this QP (matching enhancements between the requestor and the responder)
    // Store this in our own qpInfo for returning to the requestor
    if (remMeta.qpInfo[q].ece_supported && meta.qpInfo[q].ece_supported) {
      NCCLCHECKGOTO(wrap_ibv_query_ece(qp->qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    }

    // Set the backup ece (enhanced connection establishment) on this QP before RTR
    if (ncclParamEnableFaultTolerance()) {
      if (remMeta.backupQpInfo[q].ece_supported) {
        // Coverity suspects a copy-paste error below due to the use of remMeta in one argument and meta in another.
        // However, this has been confirmed to be intentional.
        // coverity[copy_paste_error]
        NCCLCHECKGOTO(wrap_ibv_set_ece(backupQp->qp, &remMeta.backupQpInfo[q].ece, &meta.backupQpInfo[q].ece_supported), ret, fail);
      }
      else {
        meta.backupQpInfo[q].ece_supported = 0;
      }

      NCCLCHECKGOTO(ncclIbRtrQp(backupQp->qp, &backupRCommDev->base.gidInfo, remMeta.backupQpInfo[q].qpn, backupRemDevInfo, true, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(backupQp->qp), ret, fail);

      // Query the reduced ece for this QP (matching enhancements between the requestor and the responder)
      if (remMeta.backupQpInfo[q].ece_supported && meta.backupQpInfo[q].ece_supported) {
        NCCLCHECKGOTO(wrap_ibv_query_ece(backupQp->qp, &meta.backupQpInfo[q].ece, &meta.backupQpInfo[q].ece_supported), ret, fail);
      }
    }

    memcpy(&qp->gidInfo, &rCommDev->base.gidInfo, sizeof(struct ncclIbGidInfo));
    qp->dest_qp_num = remMeta.qpInfo[q].qpn;
    memcpy(&qp->info, remDevInfo, sizeof(struct ncclIbDevInfo));
    qp->ece_supported = remMeta.qpInfo[q].ece_supported;
    qp->ece = remMeta.qpInfo[q].ece;
    qp->tc = remMeta.tc;
    qp->sl = remMeta.sl;
  }

  rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;
  if (ncclParamEnableFaultTolerance()) rComm->backupFlushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev ^ 1) == ncclSuccess) 
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = ncclIbDevs + rCommDev->base.ibDevN;

    if (ncclParamEnableFaultTolerance()) {
      backupRCommDev = rComm->backupDevs + i;
      backupIbDev = ncclIbDevs + backupRCommDev->base.ibDevN;
    }

    // Retain remote fifo info and prepare my RDMA ops
    rComm->remFifo.addr = remMeta.fifoAddr;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

    if (ncclParamEnableFaultTolerance()) {
      // backup Retain remote fifo info and prepare my RDMA ops
      NCCLCHECK(wrap_ibv_reg_mr(&backupRCommDev->fifoMr, backupRCommDev->base.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo) * MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
      backupRCommDev->fifoSge.lkey = backupRCommDev->fifoMr->lkey;

      // Retain remote sync fifo info and prepare my RDMA ops
      rComm->remSyncFifo.addr = remMeta.syncFifoAddr;
      NCCLCHECK(wrap_ibv_reg_mr(&rCommDev->syncFifoMr, rCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(struct ncclIbSyncFifo) * MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
      rCommDev->syncFifoSge.lkey = rCommDev->syncFifoMr->lkey;

      // backup Retain remote sync fifo info and prepare my RDMA ops
      NCCLCHECK(wrap_ibv_reg_mr(&backupRCommDev->syncFifoMr, backupRCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(struct ncclIbSyncFifo) * MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
      backupRCommDev->syncFifoSge.lkey = backupRCommDev->syncFifoMr->lkey;
    }

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE), ret, fail);
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->base.stats, &rCommDev->gpuFlush.qp, false), ret, fail);
      struct ncclIbDevInfo devInfo;
      devInfo.lid         = ibDev->portAttr.lid;
      devInfo.link_layer  = ibDev->portAttr.link_layer;
      devInfo.ib_port     = ibDev->portNum;
      devInfo.gid.global.subnet_prefix        = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.gid.global.interface_id         = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu         = ibDev->portAttr.active_mtu;
      NCCLCHECKGOTO(ncclIbRtrQp(rCommDev->gpuFlush.qp.qp, &rCommDev->base.gidInfo, rCommDev->gpuFlush.qp.qp->qp_num, &devInfo, false, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(rCommDev->gpuFlush.qp.qp), ret, fail);

      *(u_int *)rCommDev->gpuFlush.qp.srcIp = *(u_int *)(&rCommDev->base.gidInfo.localGid.raw[12]);
      *(u_int *)rCommDev->gpuFlush.qp.dscIp = *(u_int *)(&rCommDev->base.gidInfo.localGid.raw[12]);
    }

    // backup Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->backupFlushEnabled && ncclParamEnableFaultTolerance()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupRCommDev->gpuFlush.hostMr, backupRCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE), ret, fail);
      backupRCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      backupRCommDev->gpuFlush.sge.length = 1;
      backupRCommDev->gpuFlush.sge.lkey = backupRCommDev->gpuFlush.hostMr->lkey;
      NCCLCHECKGOTO(ncclIbCreateQp(backupIbDev->portNum, &backupRCommDev->base, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->base.backupStats, &backupRCommDev->gpuFlush.qp, true), ret, fail);
      struct ncclIbDevInfo devInfo;
      devInfo.lid = backupIbDev->portAttr.lid;
      devInfo.link_layer = backupIbDev->portAttr.link_layer;
      devInfo.ib_port = backupIbDev->portNum;
      devInfo.gid.global.subnet_prefix = backupRCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.gid.global.interface_id = backupRCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu = backupIbDev->portAttr.active_mtu;
      NCCLCHECKGOTO(ncclIbRtrQp(backupRCommDev->gpuFlush.qp.qp, &backupRCommDev->base.gidInfo, backupRCommDev->gpuFlush.qp.qp->qp_num, &devInfo, false, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(backupRCommDev->gpuFlush.qp.qp), ret, fail);
      *(u_int *)backupRCommDev->gpuFlush.qp.srcIp = *(u_int *)(&backupRCommDev->base.gidInfo.localGid.raw[12]);
      *(u_int *)backupRCommDev->gpuFlush.qp.dscIp = *(u_int *)(&backupRCommDev->base.gidInfo.localGid.raw[12]);
    }

    // Fill Handle
    meta.devs[i].lid                            = ibDev->portAttr.lid;
    meta.devs[i].link_layer                     = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port                        = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix       = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id        = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu                            = ibDev->portAttr.active_mtu;

    // backup Fill Handle
    if (ncclParamEnableFaultTolerance()) {
      meta.backupDevs[i].lid                      = backupIbDev->portAttr.lid;
      meta.backupDevs[i].link_layer               = backupRCommDev->base.gidInfo.link_layer = backupIbDev->portAttr.link_layer;
      meta.backupDevs[i].ib_port                  = backupIbDev->portNum;
      meta.backupDevs[i].gid.global.subnet_prefix = backupRCommDev->base.gidInfo.localGid.global.subnet_prefix;
      meta.backupDevs[i].gid.global.interface_id  = backupRCommDev->base.gidInfo.localGid.global.interface_id;
      meta.backupDevs[i].mtu                      = backupIbDev->portAttr.active_mtu;
    }

    // Prepare sizes fifo
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;

    if (ncclParamEnableFaultTolerance()) {
      // backup Prepare sizes fifo
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->backupDevs[i].sizesFifoMr, rComm->backupDevs[i].base.pd, rComm->sizesFifo, sizeof(int) * MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      meta.backupDevs[i].fifoRkey = rComm->backupDevs[i].sizesFifoMr->rkey;
    }
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn      = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;

    if (ncclParamEnableFaultTolerance()) {
      meta.backupQpInfo[q].qpn      = rComm->base.backupQps[q].qp->qp_num;
      meta.backupQpInfo[q].devIndex = rComm->base.backupQps[q].devIndex;
    }
  }
  meta.ndevs = rComm->base.vProps.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);
  if (ncclParamEnableFaultTolerance()) strncpy(meta.backupDevName, backupMergedDev->devName, MAX_MERGED_DEV_NAME);
  rComm->base.nDataQps = std::max(rComm->base.vProps.ndevs, rComm->base.nRemDevs);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));

  for (int q = 0; q < rComm->base.nqps; q++) {
    *(u_int *)rComm->base.qps[q].srcIp = *(u_int *)(&rComm->devs[rComm->base.qps[q].devIndex].base.gidInfo.localGid.raw[12]);

    if (ncclParamEnableFaultTolerance()) *(u_int *)rComm->base.backupQps[q].srcIp = *(u_int *)(&rComm->backupDevs[rComm->base.backupQps[q].devIndex].base.gidInfo.localGid.raw[12]);
  }
  for (int q = 0; q < rComm->base.nqps; q++) {
    struct ncclIbQpInfo *remQpInfo = remMeta.qpInfo + q;
    // struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;
    *(u_int *)rComm->base.qps[q].dscIp = *(u_int *)(&rComm->base.remDevs[remQpInfo->devIndex].remoteGid.raw[12]);

    if (ncclParamEnableFaultTolerance()) {
      struct ncclIbQpInfo *remBackupQpInfo = remMeta.backupQpInfo + q;
      *(u_int *)rComm->base.backupQps[q].dscIp = *(u_int *)(&rComm->base.backupRemDevs[remBackupQpInfo->devIndex].remoteGid.raw[12]);
    }
  }

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *recvComm = rComm;
exit:
  /* reset lComm stage */
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return ret;
fail:
  free(rComm);
  goto exit;
}

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      memset(r->devBases, 0, sizeof(r->devBases));
      memset(r->backupDevBases, 0, sizeof(r->backupDevBases));
      memset(r->events, 0, sizeof(r->events));
      r->time = get_nanoseconds();
      r->time_out = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  r->time = 0;
  r->time_out = 0;
  return ncclSuccess;
}

ncclResult_t _ncclIbFreeRequest(void *r) {
  return ncclIbFreeRequest((struct ncclIbRequest *)r);
}

ncclResult_t ncclIbTest(void* request, int* done, int* size);

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC;
      bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags));
        } else {
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
      } else {
        if (relaxedOrdering) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags));
        }
        else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags));
        }
      }
      TRACE(NCCL_INIT|NCCL_NET,"regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr, (long long)pages*pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population) memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
        ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

struct ncclIbNetCommDevBase *ncclIbGetBackupNetCommDevBase(ncclIbNetCommBase *base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm *sComm = (struct ncclIbSendComm *)base;
    return &sComm->backupDevs[devIndex].base;
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)base;
    return &rComm->backupDevs[devIndex].base;
  }
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i), ret, fail);

    // fill backup mhandleWrapper->mrs
    if (ncclParamEnableFaultTolerance()) {
      struct ncclIbNetCommDevBase *backupDevComm = ncclIbGetBackupNetCommDevBase(base, i);
      NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(backupDevComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i + NCCL_IB_MAX_DEVS_PER_NIC), ret, fail);
    }
  }
  *mhandle = (void*) mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);
}
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));

    // fill backup mhandleWrapper->mrs
    if (ncclParamEnableFaultTolerance()) {
      struct ncclIbNetCommDevBase *backupDevComm = ncclIbGetBackupNetCommDevBase(base, i);
      NCCLCHECK(ncclIbDeregMrInternal(backupDevComm, mhandleWrapper->mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]));
    }
  }
  free(mhandleWrapper);
  return ncclSuccess;
}

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

// count the number of send wrs and remain data sizes in wrs
thread_local int sendWrCounter[NCCL_IB_MAX_DEVS_PER_NIC] = {0};
thread_local int remainWrDataSize[NCCL_IB_MAX_DEVS_PER_NIC] = {0};

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);
#ifdef NCCL_ENABLE_NET_PROFILING
    reqs[r]->pInfo[0].nEventHandles = 0;
#endif
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    int* sizes = comm->remSizesFifo.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs*sizeof(int);
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr = comm->remSizesFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;

  for (int r = 0; r < nreqs; r++) {
    for (int q = 0; q < comm->base.vProps.ndevs; q++) {
      if (global_timer_log.collect) {
        reqs[r]->log[q].loged_start = NCCL_LOG_TELEMETRY;
        clock_gettime(CLOCK_REALTIME, &reqs[r]->log[q].send_start);
        reqs[r]->lTest[q].status = LINK_STATUS_UNUSED;
        // reqs[r]->log[q].size = reqs[r]->send.size;
        reqs[r]->log[q].size = 0;
      }
      else
        reqs[r]->log[q].loged_start = NCCL_LOG_NOT_USE;
    }
  }

  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;

    // check if qp is available
    bool if_backup = false;
    if (ncclParamEnableFaultTolerance() && comm->devs[devIndex].base.warn.is_warn == true) {
      qp = comm->base.backupQps + qpIndex;
      devIndex = qp->devIndex;
      if_backup = true;
    }

    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // Select proper rkey (needed even for 0-size send)
      // update sendWrCounter
      sendWrCounter[devIndex]++;
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = std::min(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
      reqs[r]->log[devIndex].sendWrCounter = sendWrCounter[devIndex];
      if (length > 0)
        reqs[r]->log[devIndex].size += length;

      // update remainWrDataSize
      if (length > 0) {
        remainWrDataSize[devIndex] += length;
      }
      reqs[r]->log[devIndex].remainWrDataSize = remainWrDataSize[devIndex];
    }

    if (nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      if (!if_backup) {
        comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
        lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
      }
      else {
        comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex + NCCL_IB_MAX_DEVS_PER_NIC]->lkey;
        lastWr->wr.rdma.rkey = comm->remSizesFifo.backupRkeys[devIndex];
      }
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store comm qpIndex for this request
      int nEventHandles = reqs[r]->pInfo[0].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      reqs[r]->pInfo[0].qpIndex[nEventHandles] = qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      reqs[r]->pInfo[0].data.type = ncclProfileQp;
      reqs[r]->pInfo[0].data.qp.device = devIndex;
      reqs[r]->pInfo[0].data.qp.wr_id = comm->wrs[r].wr_id;
      reqs[r]->pInfo[0].data.qp.opcode = comm->wrs[r].opcode;
      reqs[r]->pInfo[0].data.qp.qpNum = qp->qp->qp_num;
      reqs[r]->pInfo[0].data.qp.length = comm->sges[r].length;
      void* pHandle = reqs[r]->pInfo[0].pHandle;
      NCCLCHECK(ncclProfilerFunction(&reqs[r]->pInfo[0].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, pHandle, pluginId, &reqs[r]->pInfo[0].data));
      reqs[r]->pInfo[0].nEventHandles++;
    }
#endif
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // choose normal qp or backup qp according to the backup flag
    // we ensure use both normal or backup qp in dual ports
    if (ncclParamEnableFaultTolerance() && slots[r].if_backup != comm->devs[0].base.warn.is_warn) {
      for (int d = 0; d < comm->base.vProps.ndevs; d++) {
        comm->devs[d].base.warn.is_warn = slots[r].if_backup;
      }
    }
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
      return ncclInternalError;
    }

    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;
#ifdef NCCL_ENABLE_NET_PROFILING
    req->pInfo[0].pHandle = phandle;
#endif

    // Populate events
    int nEvents = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      ncclIbQp* qp = comm->base.qps + qpIndex;

      bool if_backup = false;
      if (ncclParamEnableFaultTolerance() && comm->devs[qp->devIndex].base.warn.is_warn == true) {
        qp = comm->base.backupQps + qpIndex;
        if_backup = true;
      }
      int devIndex = qp->devIndex;
      // add event
      if (!if_backup) ncclIbAddEvent(req, devIndex, &comm->devs[devIndex].base, if_backup);
      else ncclIbAddEvent(req, devIndex, &comm->backupDevs[devIndex].base, if_backup);

      *(u_int *)req->log[devIndex].srcIp = *(u_int *)qp->srcIp;
      *(u_int *)req->log[devIndex].dscIp = *(u_int *)qp->dscIp;
      req->log[devIndex].channel_id = qp->channel_id;
      req->log[devIndex].rank = comm->rank;
      req->log[devIndex].func = comm->func;
      req->log[devIndex].ncclFuncTimes = comm->ncclFuncTimes;
      req->log[devIndex].peerRank = comm->peerRank;
      req->log[devIndex].groupHash = comm->groupHash;

      req->lTest[devIndex].linkPingQp = qp->qp;
      req->log[devIndex].NetworkCardName = qp->NetworkCardName;
      // Track the valid lkey for this RDMA_Write
      if (!if_backup) req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      else req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex + NCCL_IB_MAX_DEVS_PER_NIC]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same set of QPs inside ncclIbMultiSend()
      qpIndex = (qpIndex+1)%comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      bool if_backup = false;
      if (ncclParamEnableFaultTolerance() && comm->devs[i].base.warn.is_warn == true){
        if_backup = true;
      }
      if (!if_backup){
        req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
      }
      else {
        req->send.lkeys[i] = mhandleWrapper->mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]->lkey;
      }
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

// change the fifo head
ncclResult_t ncclIbChangeFifoHead(void *sendComm, uint64_t _FifoHead) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)sendComm;
  comm->fifoHead = _FifoHead;
  return ncclSuccess;
}

ncclResult_t ncclIbCheckSubSyncFifo(void *sendComm, bool &if_rollback) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)sendComm;
  volatile struct ncclIbSyncFifo *slots;

  int slot = (comm->syncFifoHead) % MAX_REQUESTS;
  slots = comm->syncFifo;
  uint64_t idx = comm->syncFifoHead + 1;
  if (slots[slot].idx == idx)
  {
    if_rollback = true;
  }
  return ncclSuccess;
}

// send check the sync fifo
ncclResult_t ncclIbCheckSyncFifo(void *sendComm, uint64_t &recvFifoTail, uint64_t &restartPos, int &errPortIdx) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)sendComm;

  // Wait for the receiver to have posted the corresponding sync fifo
  volatile struct ncclIbSyncFifo *slots;

  int slot = (comm->syncFifoHead) % MAX_REQUESTS;
  slots = comm->syncFifo;
  uint64_t idx = comm->syncFifoHead + 1;

  // Wait until the fifo is filled
  while (slots[slot].idx != idx)
    ;
  __sync_synchronize();

  // get the recvFifoTail and restartPos
  recvFifoTail = slots[slot].recvFifoTail;
  restartPos = slots[slot].restartPos;
  errPortIdx = slots[slot].errPortIdx;
  comm->devs[slots[slot].errPortIdx].base.warn.is_warn = true;
  if (comm->base.vProps.ndevs > 1)
    comm->devs[errPortIdx ^ 1].base.warn.is_warn = true;

  comm->syncFifoHead++;
  return ncclSuccess;
}

// recv post to the sync fifo
ncclResult_t ncclIbPostSyncFifo(void *recvComm, uint64_t restartPos, int errPortIdx) {
  struct ncclIbRecvComm *comm = (struct ncclIbRecvComm *)recvComm;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remSyncFifo.syncFifoTail % MAX_REQUESTS;
  struct ncclIbSyncFifo *localElem = &comm->remSyncFifo.elems[slot];

  // when come into this function, you should use backup qp
  ncclIbQp *backupCtsQp = comm->base.backupQps + comm->base.backupDevIndex;
  comm->base.backupDevIndex = (comm->base.backupDevIndex + 1) % comm->base.vProps.ndevs;

  comm->remFifo.fifoTail += 1000;

  localElem->recvFifoTail = comm->remFifo.fifoTail;
  localElem->restartPos = restartPos;
  localElem->idx = comm->remSyncFifo.syncFifoTail + 1;
  localElem->errPortIdx = errPortIdx;

  // fill wr
  wr.wr.rdma.remote_addr = comm->remSyncFifo.addr + slot * sizeof(struct ncclIbSyncFifo);
  wr.wr.rdma.rkey = comm->base.backupRemDevs[backupCtsQp->remDevIdx].syncFifoRkey;
  comm->backupDevs[backupCtsQp->devIndex].syncFifoSge.addr = (uint64_t)localElem;
  comm->backupDevs[backupCtsQp->devIndex].syncFifoSge.length = sizeof(struct ncclIbSyncFifo);
  wr.sg_list = &comm->backupDevs[backupCtsQp->devIndex].syncFifoSge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;

  // write
  struct ibv_send_wr *bad_wr;
  NCCLCHECK(wrap_ibv_post_send(backupCtsQp->qp, &wr, &bad_wr));

  comm->remSyncFifo.syncFifoTail++;

  return ncclSuccess;
}

ncclResult_t ncclIbRePostFifoInTimeout(struct ncclIbRequest *req) {
  // fill retrasition wr
  req->retransitionWr.wr.rdma.rkey = req->base->remDevs[req->retransitionDevIndex].fifoRkey;

  // get cts qp
  ncclIbQp *ctsQp = req->base->qps + req->retransitionDevIndex;
  req->retransitionDevIndex = (req->retransitionDevIndex + 1) % req->base->vProps.ndevs;

  // send cts
  struct ibv_send_wr *bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &req->retransitionWr, &bad_wr));

  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail%MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  // Select the next devIndex (local) and QP to use for posting this CTS message
  // Since QPs are initialized by striping across devIndex, we can simply assign this to the same value
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.vProps.ndevs;

  ncclIbQp *backupCtsQp = NULL;
  bool if_backup = false;
  if (ncclParamEnableFaultTolerance() && comm->devs[ctsQp->devIndex].base.warn.is_warn == true) {
    backupCtsQp = comm->base.backupQps + comm->base.backupDevIndex;
    comm->base.backupDevIndex = (comm->base.backupDevIndex + 1) % comm->base.vProps.ndevs;
    if_backup = true;
  }

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // Send all applicable rkeys
    for (int j = 0; j < comm->base.vProps.ndevs; j++) {
      if (!ncclParamEnableFaultTolerance() || !comm->devs[j].base.warn.is_warn)
        localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
      else
        localElem[i].rkeys[j] = mhandleWrapper->mrs[j + NCCL_IB_MAX_DEVS_PER_NIC]->rkey;
    }

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].if_backup = if_backup;
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }
  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct fifoRkey
  if (!if_backup) {
    wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;
  }
  else {
    wr.wr.rdma.rkey = comm->base.backupRemDevs[backupCtsQp->remDevIdx].fifoRkey;
  }

  // Set the correct sge properties
  if (!if_backup) {
    comm->devs[ctsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
    comm->devs[ctsQp->devIndex].fifoSge.length = n * sizeof(struct ncclIbSendFifo);
    wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  }
  else {
    comm->backupDevs[backupCtsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
    comm->backupDevs[backupCtsQp->devIndex].fifoSge.length = n * sizeof(struct ncclIbSendFifo);
    wr.sg_list = &comm->backupDevs[backupCtsQp->devIndex].fifoSge;
  }

  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE

  // prepare retrasition wr for fault tolerance
  memset(&req->retransitionWr, 0, sizeof(req->retransitionWr));
  req->retransitionElem = &localElem[n - 1];

  // retransition wr only use normal qp
  req->retransitionDevIndex = comm->base.devIndex;
  req->retransitionWr.wr.rdma.remote_addr = comm->remFifo.addr + slot * NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbSendFifo) + sizeof(struct ncclIbSendFifo) * (n - 1);
  req->retransitionSge.addr = (uint64_t)req->retransitionElem;
  req->retransitionSge.length = sizeof(struct ncclIbSendFifo);
  req->retransitionSge.lkey = comm->devs[req->retransitionDevIndex].fifoSge.lkey;
  req->retransitionWr.sg_list = &req->retransitionSge;
  req->retransitionWr.num_sge = 1;
  req->retransitionWr.opcode = IBV_WR_RDMA_WRITE;
  req->retransitionWr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE
  req->retransitionWr.wr_id = req - comm->base.reqs;

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to fifo slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each fifo posting QP gets drained
  if (!if_backup) {
    if (((comm->base.vProps.ndevs == 1) && (slot == 0)) ||
        ((comm->base.vProps.ndevs > 1) && (slot == 0 || slot == 1))) {
      wr.send_flags |= IBV_SEND_SIGNALED;
      wr.wr_id = req - comm->base.reqs;
      ncclIbAddEvent(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base, if_backup);

      *(u_int *)req->log[ctsQp->devIndex].srcIp = *(u_int *)ctsQp->srcIp;
      *(u_int *)req->log[ctsQp->devIndex].dscIp = *(u_int *)ctsQp->dscIp;
      req->lTest[ctsQp->devIndex].linkPingQp = ctsQp->qp;

      if (global_timer_log.collect) {
        req->log[ctsQp->devIndex].loged_start = NCCL_LOG_TELEMETRY;
        req->lTest[ctsQp->devIndex].status = LINK_STATUS_UNUSED;
        req->log[ctsQp->devIndex].size = n * sizeof(struct ncclIbSendFifo);
        clock_gettime(CLOCK_REALTIME, &req->log[ctsQp->devIndex].send_start);
      }
      else
        req->log[ctsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
    else {
      req->log[ctsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
  }
  else {
    if (((comm->base.vProps.ndevs == 1) && (slot == 0)) ||
        ((comm->base.vProps.ndevs > 1) && (slot == 0 || slot == 1))) {
      wr.send_flags |= IBV_SEND_SIGNALED;
      wr.wr_id = req - comm->base.reqs;
      ncclIbAddEvent(req, backupCtsQp->devIndex, &comm->backupDevs[backupCtsQp->devIndex].base, if_backup);

      *(u_int *)req->log[backupCtsQp->devIndex].srcIp = *(u_int *)backupCtsQp->srcIp;
      *(u_int *)req->log[backupCtsQp->devIndex].dscIp = *(u_int *)backupCtsQp->dscIp;
      req->lTest[backupCtsQp->devIndex].linkPingQp = backupCtsQp->qp;

      if (global_timer_log.collect) {
        req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_TELEMETRY;
        req->lTest[backupCtsQp->devIndex].status = LINK_STATUS_UNUSED;
        req->log[backupCtsQp->devIndex].size = n * sizeof(struct ncclIbSendFifo);
        clock_gettime(CLOCK_REALTIME, &req->log[backupCtsQp->devIndex].send_start);
      }
      else
        req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
    else {
      req->log[backupCtsQp->devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
  }

  struct ibv_send_wr* bad_wr;
  if (!if_backup) {
    NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  }
  else {
    NCCLCHECK(wrap_ibv_post_send(backupCtsQp->qp, &wr, &bad_wr));
  }
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0");
    *request = NULL;
    return ncclInternalError;
  }
  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
#ifdef NCCL_ENABLE_NET_PROFILING
  for (int r = 0; r < n && phandles; r++) req->pInfo[r].nEventHandles = 0;
#endif

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
    if (ncclParamEnableFaultTolerance()) req->backupDevBases[i] = &comm->backupDevs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    bool if_backup = false;
    // check if qp is available
    if (ncclParamEnableFaultTolerance() && comm->devs[qp->devIndex].base.warn.is_warn == true) {
      if_backup = true;
      qp = comm->base.backupQps + comm->base.qpIndex;
      ncclIbAddEvent(req, qp->devIndex, &comm->backupDevs[qp->devIndex].base, if_backup);
    }
    else {
      ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base, if_backup);
    }

    *(u_int *)req->log[qp->devIndex].srcIp = *(u_int *)qp->srcIp;
    *(u_int *)req->log[qp->devIndex].dscIp = *(u_int *)qp->dscIp;
    req->lTest[qp->devIndex].linkPingQp = qp->qp;
    if (global_timer_log.collect) {
      clock_gettime(CLOCK_REALTIME, &req->log[qp->devIndex].send_start);
      req->lTest[qp->devIndex].status = LINK_STATUS_UNUSED;
      req->log[qp->devIndex].loged_start = NCCL_LOG_TELEMETRY;
    }
    else
      req->log[qp->devIndex].loged_start = NCCL_LOG_NOT_USE;
#ifdef NCCL_ENABLE_NET_PROFILING
    // Start a QP event for every request in the multirecv and every qp
    for (int r = 0; r < n; r++) {
      int nEventHandles = req->pInfo[r].nEventHandles;
      assert(nEventHandles < MAX_QPS_PER_REQ);
      req->pInfo[r].qpIndex[nEventHandles] = comm->base.qpIndex;
      // Store info for profiler
      int64_t pluginId = NCCL_PROFILER_NET_TYPE_IB | NCCL_PROFILER_NET_IB_VER;
      req->pInfo[r].data.type = ncclProfileQp;
      req->pInfo[r].data.qp.device = qp->devIndex;
      req->pInfo[r].data.qp.wr_id = wr.wr_id;
      req->pInfo[r].data.qp.qpNum = qp->qp->qp_num;
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[r].qpEventHandles[nEventHandles], ncclProfilerNetEventStart, phandles[r], pluginId, &req->pInfo[r].data));
      req->pInfo[r].nEventHandles++;
    }
#endif
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1)%comm->base.nqps;
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    bool if_backup = false;
    if (ncclParamEnableFaultTolerance() && comm->devs[i].base.warn.is_warn == true) {
      if_backup = true;
      wr.wr.rdma.rkey = mhandle->mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]->rkey;
      wr.sg_list = &comm->backupDevs[i].gpuFlush.sge;
    }
    else {
      wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
      wr.sg_list = &comm->devs[i].gpuFlush.sge;
    }

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    if (!if_backup) {
      *(u_int *)req->log[comm->devs[i].gpuFlush.qp.devIndex].srcIp = *(u_int *)comm->devs[i].gpuFlush.qp.srcIp;
      *(u_int *)req->log[comm->devs[i].gpuFlush.qp.devIndex].dscIp = *(u_int *)comm->devs[i].gpuFlush.qp.dscIp;
      if (global_timer_log.collect) {
        req->log[comm->devs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_TELEMETRY;
        req->lTest[comm->devs[i].gpuFlush.qp.devIndex].status = LINK_STATUS_UNUSED;

        clock_gettime(CLOCK_REALTIME, &req->log[comm->devs[i].gpuFlush.qp.devIndex].send_start);
        // req->log[comm->devs[i].gpuFlush.qp.devIndex].size = 0;
      }
      else
        req->log[comm->devs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_NOT_USE;
    }
    else {
      *(u_int *)req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].srcIp = *(u_int *)comm->backupDevs[i].gpuFlush.qp.srcIp;
      *(u_int *)req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].dscIp = *(u_int *)comm->backupDevs[i].gpuFlush.qp.dscIp;
      if (global_timer_log.collect) {
        req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_TELEMETRY;
        req->lTest[comm->backupDevs[i].gpuFlush.qp.devIndex].status = LINK_STATUS_UNUSED;

        clock_gettime(CLOCK_REALTIME, &req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].send_start);
      }
      else
        req->log[comm->backupDevs[i].gpuFlush.qp.devIndex].loged_start = NCCL_LOG_NOT_USE;
    }

    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    if (!if_backup) {
      NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    }
    else {
      NCCLCHECK(wrap_ibv_post_send(comm->backupDevs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    }
    TIME_STOP(4);

    if (!if_backup) ncclIbAddEvent(req, i, &comm->devs[i].base, if_backup);
    else ncclIbAddEvent(req, i, &comm->backupDevs[i].base, if_backup);
  }

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

#ifdef NCCL_ENABLE_NET_PROFILING
static int getReqQpIndex(struct ncclIbRequest* req, int request, int qpNumber) {
  for (int i = 0; i < MAX_QPS_PER_REQ; i++) {
    int qpIndex = req->pInfo[request].qpIndex[i];
    if (req->base->qps[qpIndex].qp->qp_num == qpNumber) return i;
  }
  return 0;
}
#endif

ncclResult_t getLinkStatus(struct ncclIbRequest *r, int i, int *status) {
  /*linkStatusPing*/
  if (r->events[i] == 0) {
    *status = r->lTest[i].status = LINK_STATUS_SUCCESS;
    return ncclSuccess;
  }
  if (r->lTest[i].status == LINK_STATUS_UNUSED) {
    if (r->lTest[i].linkPingQp) {
      if (r->type == NCCL_NET_IB_REQ_SEND)
        r->events[i] += r->nreqs;
      else if (r->type == NCCL_NET_IB_REQ_RECV)
        r->events[i]++;
      else {
        r->lTest[i].events = r->events[i];
        *status = r->lTest[i].status = LINK_STATUS_WRONG;
        return ncclSuccess;
      }
      struct ibv_send_wr wr, *bad_wr;
      memset(&wr, 0, sizeof(wr));
      for (int j = 0; j < r->nreqs; j++)
        wr.wr_id |= (r - r->base->reqs) << (j * 8);
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      r->lTest[i].events = r->events[i];
      clock_gettime(CLOCK_REALTIME, &r->lTest[i].send_start);
      r->lTest[i].status = LINK_STATUS_USED;
      NCCLCHECK(wrap_ibv_post_send(r->lTest[i].linkPingQp, &wr, &bad_wr));
    }
    else
      r->lTest[i].status = LINK_STATUS_WRONG;
  }
  else if (r->lTest[i].status == LINK_STATUS_USED) {
    clock_gettime(CLOCK_REALTIME, &r->lTest[i].send_end);
    if (r->lTest[i].send_end.tv_sec - r->lTest[i].send_start.tv_sec >= 3 && r->events[i] == r->lTest[i].events)
      r->lTest[i].status = LINK_STATUS_WRONG;
    else if (r->events[i] < r->lTest[i].events) {
      r->lTest[i].status = LINK_STATUS_SUCCESS;
    }
  }
  else if (r->lTest[i].status == LINK_STATUS_WRONG) {
    if (r->events[i] < r->lTest[i].events) {
      r->lTest[i].status = LINK_STATUS_SUCCESS;
    }
    else {
      clock_gettime(CLOCK_REALTIME, &r->lTest[i].send_end);
      r->lTest[i].status = LINK_STATUS_WRONG_WAIT;
    }
  }
  else if (r->lTest[i].status == LINK_STATUS_WRONG_WAIT) {
    if (r->events[i] < r->lTest[i].events) {
      r->lTest[i].status = LINK_STATUS_SUCCESS;
    }
    else {
      struct timespec cur_time;
      clock_gettime(CLOCK_REALTIME, &cur_time);
      if (cur_time.tv_sec - r->lTest[i].send_end.tv_sec >= 3)
        r->lTest[i].status = LINK_STATUS_WRONG;
    }
  }
  // else if(r->lTest[i].status == LINK_STATUS_SUCCESS){
  //   ;
  // }
  *status = r->lTest[i].status;
  return ncclSuccess;
}

void ncclIbTestOutput(struct ncclIbRequest *r) {
  for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
    if (r->events[i] && r->devBases[i]->warn.is_warn) {
      if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        WARN("NET/IB : Got completion from peer %s with status=%d opcode=%d len=%d vendor err %d (%s)%s%s%s%s",
             r->devBases[i]->warn.line.c_str(), r->devBases[i]->warn.status, r->devBases[i]->warn.opcode, r->devBases[i]->warn.len, r->devBases[i]->warn.error, r->devBases[i]->warn.type.c_str(),
             r->devBases[i]->warn.localGidstr.c_str(), r->devBases[i]->warn.localGidstring.c_str(), r->devBases[i]->warn.remoteGidstr.c_str(), r->devBases[i]->warn.remoteGidstring.c_str());
      }
      else {
        WARN("NET/IB : Got completion from peer %s with status=%d opcode=%d len=%d vendor err %d (%s)",
             r->devBases[i]->warn.line.c_str(), r->devBases[i]->warn.status, r->devBases[i]->warn.opcode, r->devBases[i]->warn.len, r->devBases[i]->warn.error, r->devBases[i]->warn.type.c_str());
      }
    }
  }
}

ncclResult_t ncclIbGetErrorPortIdx(void *request, int &idx) {
  struct ncclIbRequest *r = (struct ncclIbRequest *)request;
  if (r->devBases[0]->warn.is_warn) {
    idx = 0;
  }
  else
    idx = 1;
  return ncclSuccess;
}

ncclResult_t ncclIbGetReqSize(void *request, int *sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest *)request;
  if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
    for (int i = 0; i < r->nreqs; i++)
      sizes[i] = r->recv.sizes[i];
  }
  if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
    sizes[0] = r->send.size;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbResetQpIndex(void *comm, bool if_send) {
  // You should reset the qpIndex to 0 to avoid using different qp in round-robin mode
  if (comm == NULL)
    return ncclSuccess;

  if (if_send) {
    struct ncclIbSendComm *lComm = (struct ncclIbSendComm *)comm;
    lComm->base.qpIndex = 0;
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)comm;
    rComm->base.qpIndex = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbReTransitionQpAndCq(void *comm, bool if_send) {
  if (comm == NULL)
    return ncclSuccess;

  // use to guarantee that in dual ports condition, if one port is down, we also retransition another port
  bool ifportup = true;
  if (if_send) {
    struct ncclIbSendComm *lComm = (struct ncclIbSendComm *)comm;
    for (int j = 0; j < lComm->base.nqps && j < 2; j++) {
      if (lComm->base.qps[j].qp == NULL)
        continue;

      struct ibv_port_attr portAttr;
      struct ncclIbDev *ibdev = ncclIbDevs + (lComm->devs[lComm->base.qps[j].devIndex].base.ibDevN);
      struct ibv_context *context = ibdev->context;
      if (ncclSuccess != wrap_ibv_query_port(context, lComm->base.qps[j].ib_port, &portAttr)) {
        WARN("NET/IB : Unable to query port_num %d", lComm->base.qps[j].ib_port);
        continue;
      }
      if (portAttr.state != IBV_PORT_ACTIVE) {
        ifportup = false;
        break;
      }
    }

    for (int j = 0; j < lComm->base.nqps; j++) {
      if (lComm->base.qps[j].qp == NULL ||
          !lComm->devs[lComm->base.qps[j].devIndex].base.warn.is_warn)
        continue;

      if (!ifportup) {
        // TO avoid the case that when port is not up, the qp state is not reset
        struct ibv_qp *qp = lComm->base.qps[j].qp;
        struct ibv_qp_attr qpAttr;
        memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
        qpAttr.qp_state = IBV_QPS_ERR;
        NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE));
        continue;
      }

      // transition the qp state to reset, init, rtr, rts
      struct ibv_qp *qp = lComm->base.qps[j].qp;
      struct ibv_qp_attr qpAttr;
      memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
      qpAttr.qp_state = IBV_QPS_RESET;
      NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE));

      qpAttr.qp_state = IBV_QPS_INIT;
      qpAttr.pkey_index = 0;
      qpAttr.port_num = lComm->base.qps[j].ib_port;
      qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
      NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

      if (lComm->base.qps[j].ece_supported)
        NCCLCHECK(wrap_ibv_set_ece(qp, &lComm->base.qps[j].ece, &lComm->base.qps[j].ece_supported));

      NCCLCHECK(ncclIbRtrQp(qp, &lComm->base.qps[j].gidInfo, lComm->base.qps[j].dest_qp_num, &lComm->base.qps[j].info, false, lComm->base.qps[j].tc, lComm->base.qps[j].sl));

      NCCLCHECK(ncclIbRtsQp(qp));
    }

    // Clear all the wc
    struct ibv_wc wcs[4];
    int wrDone = 0;
    for (int i = 0; i < 2; i++) {
      if (lComm->devs[i].base.cq == NULL)
        continue;
      NCCLCHECK(wrap_ibv_poll_cq(lComm->devs[i].base.cq, 4, wcs, &wrDone));
      while (wrDone != 0) {
        NCCLCHECK(wrap_ibv_poll_cq(lComm->devs[i].base.cq, 4, wcs, &wrDone));
      }
    }

    lComm->sendCcCnt = 0;
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)comm;
    for (int j = 0; j < rComm->base.nqps && j < 2; j++) {
      if (rComm->base.qps[j].qp == NULL)
        continue;

      struct ibv_port_attr portAttr;
      struct ncclIbDev *ibdev = ncclIbDevs + (rComm->devs[rComm->base.qps[j].devIndex].base.ibDevN);
      struct ibv_context *context = ibdev->context;
      if (ncclSuccess != wrap_ibv_query_port(context, rComm->base.qps[j].ib_port, &portAttr)) {
        WARN("NET/IB : Unable to query port_num %d", rComm->base.qps[j].ib_port);
        continue;
      }
      if (portAttr.state != IBV_PORT_ACTIVE) {
        ifportup = false;
        break;
      }
    }

    for (int j = 0; j < rComm->base.nqps; j++) {
      if (rComm->base.qps[j].qp == NULL ||
          !rComm->devs[rComm->base.qps[j].devIndex].base.warn.is_warn)
        continue;

      if (!ifportup) {
        // TO avoid the case that when port is not up, the qp state is not reset
        struct ibv_qp *qp = rComm->base.qps[j].qp;
        struct ibv_qp_attr qpAttr;
        memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
        qpAttr.qp_state = IBV_QPS_ERR;
        NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE));
        continue;
      }

      // transition the qp state to reset, init, rtr, rts
      struct ibv_qp *qp = rComm->base.qps[j].qp;
      struct ibv_qp_attr qpAttr;
      memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
      qpAttr.qp_state = IBV_QPS_RESET;
      NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE));

      qpAttr.qp_state = IBV_QPS_INIT;
      qpAttr.pkey_index = 0;
      qpAttr.port_num = rComm->base.qps[j].ib_port;
      qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
      NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

      if (rComm->base.qps[j].ece_supported)
        NCCLCHECK(wrap_ibv_set_ece(qp, &rComm->base.qps[j].ece, &rComm->base.qps[j].ece_supported));

      NCCLCHECK(ncclIbRtrQp(qp, &rComm->base.qps[j].gidInfo, rComm->base.qps[j].dest_qp_num, &rComm->base.qps[j].info, true, rComm->base.qps[j].tc, rComm->base.qps[j].sl));

      NCCLCHECK(ncclIbRtsQp(qp));
    }

    // Clear all the wc
    struct ibv_wc wcs[4];
    int wrDone = 0;
    for (int i = 0; i < 2; i++) {
      if (rComm->devs[i].base.cq == NULL)
        continue;
      NCCLCHECK(wrap_ibv_poll_cq(rComm->devs[i].base.cq, 4, wcs, &wrDone));
      while (wrDone != 0) {
        NCCLCHECK(wrap_ibv_poll_cq(rComm->devs[i].base.cq, 4, wcs, &wrDone));
      }
    }

    rComm->recvCcCnt = 0;
  }

  return ncclSuccess;
}

ncclResult_t ncclIbCheckIfNeedStreamSync(void *comm, bool if_send, bool &needStreamSync) {
  if (comm == NULL)
    return ncclSuccess;
  // Reset qp and clear wr
  if (if_send) {
    struct ncclIbSendComm *lComm = (struct ncclIbSendComm *)comm;

    for (int i = 0; i < lComm->base.vProps.ndevs; i++) {
      if (lComm->devs[i].base.warn.is_warn) {
        needStreamSync = true;
        break;
      }
    }
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)comm;

    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      if (rComm->devs[i].base.warn.is_warn) {
        needStreamSync = true;
        break;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclIbRefreshDevState(void *comm, bool if_send, bool ifsendrecv) {
  if (comm == NULL)
    return ncclSuccess;
  int refreshNum = (ifsendrecv ? 256 : 64);
  // Reset qp and clear wr
  if (if_send) {
    struct ncclIbSendComm *lComm = (struct ncclIbSendComm *)comm;

    // refresh the dev state
    lComm->sendCcCnt = (lComm->sendCcCnt + 1) % refreshNum;
    if (lComm->sendCcCnt != 0) {
      return ncclSuccess;
    }

    for (int i = 0; i < lComm->base.vProps.ndevs; i++) {
      lComm->devs[i].base.warn.is_warn = false;
    }
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)comm;

    // refresh the dev state
    rComm->recvCcCnt = (rComm->recvCcCnt + 1) % refreshNum;
    if (rComm->recvCcCnt != 0) {
      return ncclSuccess;
    }

    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      rComm->devs[i].base.warn.is_warn = false;
    }
  }

  __sync_synchronize();

  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  while (1) {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (r->events[0] == 0 && r->events[1] == 0 && r->events[2] == 0 && r->events[3] == 0) {
      TRACE(NCCL_NET, "r=%p done", r);
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) {
          sizes[i] = r->recv.sizes[i];
#ifdef NCCL_ENABLE_NET_PROFILING
          for (int j = 0; j < r->pInfo[i].nEventHandles; j++) {
            NCCLCHECK(ncclProfilerFunction(&r->pInfo[i].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
          }
#endif
        }
      }
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
#ifdef NCCL_ENABLE_NET_PROFILING
        for (int j = 0; j < r->pInfo[0].nEventHandles; j++) {
          NCCLCHECK(ncclProfilerFunction(&r->pInfo[0].qpEventHandles[j], ncclProfilerNetEventStop, NULL, 0, NULL));
        }
#endif
      }
      // Stop all remaining Qp events for this event
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int totalWrDone = 0;
    int wrDone = 0;
    int backupWrDone = 0;
    struct ibv_wc wcs[8];

    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      // If we expect any completions from this device's CQ
      if (r->events[i]) {
        if(r->devBases[i] != NULL && !r->devBases[i]->warn.is_warn) {
          NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        }
	      else if(r->backupDevBases[i] != NULL) {
          NCCLCHECK(wrap_ibv_poll_cq(r->backupDevBases[i]->backupCq, 4, wcs + wrDone, &backupWrDone));
        }
        totalWrDone += wrDone + backupWrDone;
        if (wrDone == 0 && backupWrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
        if (wrDone == 0 && backupWrDone == 0) continue;
        for (int w=0; w<wrDone + backupWrDone; w++) {
          struct ibv_wc *wc = wcs+w;
          if (wc->status != IBV_WC_SUCCESS) {
            union ncclSocketAddress addr;
            ncclSocketGetAddr(r->sock, &addr);
            char localGidString[INET6_ADDRSTRLEN] = "";
            char remoteGidString[INET6_ADDRSTRLEN] = "";
            const char* localGidStr = NULL, *remoteGidStr = NULL;
            if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
              localGidStr = ibvGetGidStr(&r->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
              remoteGidStr = ibvGetGidStr(&r->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
            }

            char line[SOCKET_NAME_MAXLEN+1];
            char *hcaName = r->devBases[i]->pd->context->device->name;
            std::string Line = ncclSocketToString(&addr, line);
            r->devBases[i]->warn.is_warn = true;
            if (r->devBases[i ^ 1] != NULL) r->devBases[i ^ 1]->warn.is_warn = true;
            r->devBases[i]->warn.line = Line;
            r->devBases[i]->warn.status = wc->status;
            r->devBases[i]->warn.opcode = wc->opcode;
            r->devBases[i]->warn.len = wc->byte_len;
            r->devBases[i]->warn.error = wc->vendor_err;
            r->devBases[i]->warn.type = reqTypeStr[r->type];
            if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
              r->devBases[i]->warn.localGidstr = localGidStr ? " localGid " : "";
              r->devBases[i]->warn.localGidstring = localGidString;
              r->devBases[i]->warn.remoteGidstr = remoteGidStr ? " remoteGids" : "";
              r->devBases[i]->warn.remoteGidstring = remoteGidString;
            }
            int reqSize = wc->byte_len;
            struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);
            if (req && req->type == NCCL_NET_IB_REQ_SEND) {
              // For Send use the request size as WC byte_len is not reliable
              reqSize = req->send.size;
            }
            WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) reqSize=%d vendor_err=%u req_type=%s%s%s%s%s hca %s",
                ncclSocketToString(&addr, line), ibvWcStatusStr(wc->status), wc->status,
                ibvWcOpcodeStr(wc->opcode), wc->opcode, reqSize, wc->vendor_err, reqTypeStr[r->type],
                localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
            ret =  ncclRemoteError;
            goto ret;
          }

          union ncclSocketAddress addr;
          ncclSocketGetAddr(r->sock, &addr);
          struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);

          #ifdef ENABLE_TRACE
          char line[SOCKET_NAME_MAXLEN+1];
          TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, i=%d",
            ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], i);
          #endif
          if (req && req->type == NCCL_NET_IB_REQ_SEND) {
            // update sendWrCounter
            sendWrCounter[i] -= req->nreqs;

            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs+((wc->wr_id >> (j*8)) & 0xff);
              if ((sendReq->events[i] <= 0)) {
                WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], i, j);
                ret =  ncclInternalError;
                goto ret;
              }
              sendReq->events[i]--;
              if (global_timer_log.collect && sendReq->log[i].loged_start == NCCL_LOG_TELEMETRY && !sendReq->events[i]) {
                clock_gettime(CLOCK_REALTIME, &sendReq->log[i].send_end);
                sendReq->log[i].diff = 1000000000L * (sendReq->log[i].send_end.tv_sec) + sendReq->log[i].send_end.tv_nsec;
                sendReq->log[i].sendWrCounter = sendWrCounter[i];
                sendReq->log[i].devIndex = i;

                // sendReq has completed transport, update remainWrDataSize
                remainWrDataSize[i] -= sendReq->log[i].size;
                sendReq->log[i].remainWrDataSize = remainWrDataSize[i];

                if (sendReq->log[i].size > 16) {
                  // save log
                  pthread_mutex_lock(&global_timer_log.lock);
                  __sync_synchronize();
                  global_timer_log.push(sendReq->log[i]);
                  pthread_mutex_unlock(&global_timer_log.lock);
                }
              }
#ifdef NCCL_ENABLE_NET_PROFILING
              // Stop Qp event for sendReq
              int qpIndex = getReqQpIndex(sendReq, j, wc->qp_num);
              NCCLCHECK(ncclProfilerFunction(&sendReq->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
#endif
            }
          } else {
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->type != NCCL_NET_IB_REQ_RECV) {
                WARN("NET/IB: wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM and req->type=%d", req->type);
                ret = ncclInternalError;
                goto ret;
              }
              if (req->nreqs == 1) {
                req->recv.sizes[0] = wc->imm_data;
              }
            }
            req->events[i]--;
#ifdef NCCL_ENABLE_NET_PROFILING
            // Stop Qp event for workFifo
            for (int j = 0; j < req->nreqs; j++) {
              int qpIndex = getReqQpIndex(req, j, wc->qp_num);
              NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
            }
#endif
          }
        }
        // Once the IB fatal event is reported in the async thread, we want to propagate this error
        // to communicator and prevent further polling to reduce error pollution.
        // NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
      }
    }

    if (TIMER_LOG_NCCL_HANG && global_timer_log.collect && totalWrDone == 0) {
      __sync_synchronize();
      for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
        if (r->events[i] && !r->log[i].loged_start) {
          r->log[i].loged_start = NCCL_LOG_HANG;
          r->lTest[i].status = LINK_STATUS_UNUSED;
          clock_gettime(CLOCK_REALTIME, &r->log[i].send_start);
        }
        if (r->events[i] && global_timer_log.collect && r->log[i].loged_start) {
          clock_gettime(CLOCK_REALTIME, &r->log[i].send_end);
          if (r->log[i].send_end.tv_sec - r->log[i].send_start.tv_sec >= 3) {
            int status = LINK_STATUS_SUCCESS;
            NCCLCHECK(getLinkStatus(r, i, &status));
            if (status == LINK_STATUS_WRONG) {
              r->log[i].diff = -1;
              printLogInfo(r->log[i]);
              // if(global_timer_log.setState(TIMER_LOG_QUEUE_WRITE)){
              //   global_timer_log.push(r->log[i]);
              //   global_timer_log.freeState();
              // }
            }
            else if (status == LINK_STATUS_SUCCESS) {
              r->lTest[i].status = LINK_STATUS_UNUSED;
            }
          }
        }
      }
    }

    // If no CQEs found on any device, return and come back later
    if (totalWrDone == 0) {
      ret = ncclSuccess;
      goto ret;
    }
  }
ret:

  long long elapsed = get_nanoseconds() - r->time;
  if (elapsed > 25 * second_to_nanoseconds) {
    if (r->devBases[0] != NULL && r->devBases[0]->warn.is_warn) return ret;
    if (r->time_out == 0 && r->type == NCCL_NET_IB_REQ_RECV) {
      r->time_out = 1;
      union ncclSocketAddress dbg_addr;
      ncclSocketGetAddr(r->sock, &dbg_addr);
      char dbg_line[SOCKET_NAME_MAXLEN + 1];
      const char *dbg_name = ncclSocketToString(&dbg_addr, dbg_line);
      int size = 0;
      const char *barrier_info = "unknown!";
      if (r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i = 0; i < r->nreqs; i++)
          size += r->recv.sizes[i];
        barrier_info = "op_recv@";
      }
      if (r->type == NCCL_NET_IB_REQ_SEND) {
        size += r->send.size;
        barrier_info = "op_send@";
      }
      if (size == 16) {
        barrier_info = "barrier?";
      }
      INFO(NCCL_INIT, "NCCL stall: r->events[0]: %d, r->events[1], %d %lld %d %d %d %p %d %s %s\n",
           r->events[0], r->events[1], elapsed, *done, ret, r->type, r, size, barrier_info, dbg_name);
      // re-send cts message to check link status
      // in this case, if link has errors, we can get wc with error status. By that, we can change to backup qp
      if (ncclParamEnableFaultTolerance()) ncclIbRePostFifoInTimeout(r);
    }
  }

  return ret;
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp != NULL) {
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
      }
      if (ncclParamEnableFaultTolerance() && comm->base.backupQps[q].qp != NULL) {
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.backupQps[q].qp));
      }
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (ncclParamEnableFaultTolerance() && commDev->syncFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->syncFifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i]));
      if (commDev->putSignalScratchpadMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
      if (ncclParamEnableFaultTolerance() && comm->remSizesFifo.mrs[i + NCCL_IB_MAX_DEVS_PER_NIC] != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base, false));

      if (ncclParamEnableFaultTolerance()) {
        struct ncclIbSendCommDev *backupCommDev = comm->backupDevs + i;
        if (backupCommDev->fifoMr != NULL)
          NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->fifoMr));
        if (backupCommDev->syncFifoMr != NULL)
          NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->syncFifoMr));
        NCCLCHECK(ncclIbDestroyBase(&backupCommDev->base, true));
      }
    }

    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++) {
      if (comm->base.qps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));
      if (ncclParamEnableFaultTolerance() && comm->base.backupQps[q].qp != NULL)
        NCCLCHECK(wrap_ibv_destroy_qp(comm->base.backupQps[q].qp));
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (ncclParamEnableFaultTolerance() && commDev->syncFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->syncFifoMr));
      if (commDev->sizesFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->sizesFifoMr));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base, false));

      if (ncclParamEnableFaultTolerance()) {
        struct ncclIbRecvCommDev *backupCommDev = comm->backupDevs + i;
        if (comm->flushEnabled) {
          if (backupCommDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(backupCommDev->gpuFlush.qp.qp));
          if (backupCommDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->gpuFlush.hostMr));
        }
        if (backupCommDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->fifoMr));
        if (backupCommDev->syncFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->syncFifoMr));
        if (backupCommDev->sizesFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(backupCommDev->sizesFifoMr));
        NCCLCHECK(ncclIbDestroyBase(&backupCommDev->base, true));
      }
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbFinalize(void* ctx) {
  free(ctx);
  return ncclIbFinalizeDevices();
}

ncclResult_t saveChannelToQp(void *netSendComm, int channel_id) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  for (int q = 0; q < comm->base.nqps; q++) {
    comm->base.qps[q].channel_id = channel_id;
    if (ncclParamEnableFaultTolerance()) comm->base.backupQps[q].channel_id = channel_id;
  }
  return ncclSuccess;
}

ncclResult_t setCommunicationAlgorithm(void *netSendComm, uint8_t func) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  comm->func = func;
  return ncclSuccess;
}

ncclResult_t setNcclFuncTimes(void *netSendComm, unsigned long long ncclFuncTimes) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  comm->ncclFuncTimes = ncclFuncTimes;
  return ncclSuccess;
}

ncclResult_t setNcclPeerRank(void *netSendComm, int rank) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  comm->peerRank = rank;
  return ncclSuccess;
}

ncclResult_t setNcclRank(void *netSendComm, int rank) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  comm->rank = rank;
  return ncclSuccess;
}

ncclResult_t setNcclGroupHash(void *netSendComm, uint64_t groupHash) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  comm->groupHash = groupHash;
  return ncclSuccess;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclIbMakeVDevice,
  ncclIbFinalize,
  ncclIbSetNetAttr,
};



/// GIN IB Plugin

#include "gin/gin_host.h"
#include "net_ib_gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  *ctx = netCommConfig;
  return ncclSuccess;
}

ncclResult_t ncclGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    ncclNetIb.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    ncclNetIb.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void *tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void **)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void *)((uintptr_t)recv_buf + i * len), (void *)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf)
    free(tmp_buf);

  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank * sizeof(int),
                               dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cComm = nullptr;
  int next;

  NCCLCHECK(ncclIbMalloc((void **)&cComm, sizeof(*cComm)));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->fullSendComm, sizeof(void *) * nranks));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->fullRecvComm, sizeof(void *) * nranks));

  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  for (int i = 0; i < nranks; i++)
  {
    int connectPeer = (cComm->rank + i) % nranks;
    int acceptPeer = (cComm->rank - i + nranks) % nranks;
    do
    {
      if (cComm->fullSendComm[connectPeer] == NULL)
        NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[connectPeer], &cComm->fullSendComm[connectPeer], NULL));
      if (cComm->fullRecvComm[acceptPeer] == NULL)
        NCCLCHECK(ncclNetIb.accept(lComm, &cComm->fullRecvComm[acceptPeer], NULL));
    } while ((cComm->fullSendComm[connectPeer] == NULL) || (cComm->fullRecvComm[acceptPeer] == NULL));
    NCCLCHECK(ncclGinIbP2PBarrier(cComm));
  }

  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  if (!cComm) return ncclSuccess;

  if (cComm->fullRecvComm) {
    for (int i=0; i<cComm->nranks; i++) {
      NCCLCHECK(ncclNetIb.closeRecv(cComm->fullRecvComm[i]));
    }
    free(cComm->fullRecvComm);
    cComm->fullRecvComm = NULL;
  }

  if (cComm->fullSendComm) {
    for (int i=0; i<cComm->nranks; i++) {
      NCCLCHECK(ncclNetIb.closeSend(cComm->fullSendComm[i]));
    }
    free(cComm->fullSendComm);
    cComm->fullSendComm = NULL;
  }

  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cComm);
  return ncclSuccess;
}

#include "gdaki/gin_host_gdaki.h"

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  NCCLCHECK(ncclGinIbInit(ctx, commId, logFunction));
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  NCCLCHECK(ncclNetIb.getProperties(ncclGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ctx, ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, int nSignals, int nCounters, void **ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, nSignals, nCounters, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm *)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL, // regMrSymDmaBuf
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};


struct ncclIbGinProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECKNOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&ginMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &ginMrHandle->mrHandle->mrs[0]->rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;

  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;

  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void *collComm, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank, void **request)
{
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[cComm->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_WRITE;
  wr.send_flags              = IBV_SEND_SIGNALED;
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)dstPtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr;  // Local buffer address
  sge.length = size;  // Size of the transfer
  sge.lkey = lkey;  // Local key

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base, true);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPutSignal(void *collComm, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle,
                                      uint32_t rank, uint64_t signalOff, void *signalMhandle,
                                      uint64_t signalValue, uint32_t signalOp, void **request)
{
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;
  struct ncclIbGinProxyMrHandle *signalMrHandle = (struct ncclIbGinProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    void *srcPtr = (void *)(srcMrHandle->base_vas[cComm->rank] + srcOff);
    void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode                  = IBV_WR_RDMA_WRITE;
    wr[0].send_flags              = 0; // We only need the CQE from the signal
    wr[0].wr_id                   = req - comm->base.reqs;
    wr[0].next                    = &wr[1];
    wr[0].wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey            = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void *signalPtr = (void *)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base, true);
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyTest(void *collComm, void *request, int *done) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  int rank = req->iput.rank;
  *done = 0;

  if (req->events[0] == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
    return ncclSuccess;
  }
  int wrDone = 0;
  struct ibv_wc wc[4];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)cComm->fullSendComm[rank];
  NCCLCHECK(wrap_ibv_poll_cq(comm->devs[0].base.cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char localGidString[INET6_ADDRSTRLEN] = "";
      char remoteGidString[INET6_ADDRSTRLEN] = "";
      const char* localGidStr = NULL, *remoteGidStr = NULL;
      if (req->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
        localGidStr = ibvGetGidStr(&req->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
        remoteGidStr = ibvGetGidStr(&req->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
      }

      char line[SOCKET_NAME_MAXLEN+1];
      char *hcaName = req->devBases[i]->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
          ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len, wc[i].vendor_err, reqTypeStr[req->type],
          localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = comm->base.reqs + wc[i].wr_id;

    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbInit,
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbConnect,
  NULL,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  NULL,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};
