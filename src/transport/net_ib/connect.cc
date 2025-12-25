/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", -1);
NCCL_PARAM(IbTc, "IB_TC", -1);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);
NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);
extern int64_t ncclParamEnableFaultTolerance();

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
};

// Structure used to hold information needed to establish the communication
// between the sender and receiver.
// The structure is populated during the connection establishment phase and
// populated by each side of the connection before being sent to the remote
// peer. The remote peer uses the information passed to it from its peer to
// create and initialize its local resources.
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbQpInfo backupQpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbDevInfo backupDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  char backupDevName[MAX_MERGED_DEV_NAME];
  // An address for a registered memory to be accessed by the peer. The address
  // can be accessed using RDMA using the key specified in ncclIbDevInfo::rkey.
  // The sender side gets in this member, from the receiver, the address of the
  // memory to which the sender writes the sizes of the data transfers that
  // the sender sends.
  // The receiver side gets in this member, from the sender, the address of the
  // memory to which the receiver writes the CTS messages.
  uint64_t addr;
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

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

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

ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
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
  qpInitAttr.cap.max_send_wr = 2 * MAX_REQUESTS + 1; // +1 for retransition wr
  qpInitAttr.cap.max_recv_wr = NET_IB_MAX_REQUESTS;
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

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

// The function creates and initializes QPs (modifies the QPs to INIT) on the
// sender side. Afterwards it populates the metadata structure, provided to the
// function (meta), with the QPs' information. Note that after the QPs'
// creation, the QPs are also queried for ECE support and the metadata structure
// is updated accordingly. The meta data structure is then expected to be
// delivered to the remote side (receiver) as part of the connection
// establishment process.
static ncclResult_t ncclIbSenderQpsCreate(ncclIbSendComm* comm, struct ncclIbConnectionMetadata* meta) {
  uint nqps = comm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % comm->base.vProps.ndevs;
    ncclIbSendCommDev* commDev = &comm->devs[devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[commDev->base.ibDevN];
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];
    int qpAccessFlags = IBV_ACCESS_REMOTE_WRITE;

    NCCLCHECK(ncclIbCreateQp(ibDev->portNum, &commDev->base, qpAccessFlags, &comm->base.stats, localQp, false));
    localQp->devIndex = devIndex;

    // Populate the metadata that will be delivered to the remote peer
    localQpInfo->qpn      = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // backup QP
    if (ncclParamEnableFaultTolerance()) {
      ncclIbSendCommDev *backupCommDev = &comm->backupDevs[devIndex];
      ncclIbDev *backupIbDev = &ncclIbDevs[backupCommDev->base.ibDevN];
      ncclIbQp *backupQp = &comm->base.backupQps[qpIndex];
      ncclIbQpInfo *backupQpInfo = &meta->backupQpInfo[qpIndex];
      NCCLCHECK(ncclIbCreateQp(backupIbDev->portNum, &backupCommDev->base, qpAccessFlags, &comm->base.backupStats, backupQp, true));
      backupQp->devIndex = devIndex;

      // Populate the backup metadata that will be delivered to the remote peer
      backupQpInfo->qpn = backupQp->qp->qp_num;
      backupQpInfo->devIndex = backupQp->devIndex;
    }

    if (ncclParamIbEceEnable()) {
      // Query ECE (Enhanced Connection Establishment) capabilities
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQpInfo->ece_supported));
      if (ncclParamEnableFaultTolerance()) {
        NCCLCHECK(wrap_ibv_query_ece(backupQp->qp, &backupQpInfo->ece, &backupQpInfo->ece_supported));
      }
    } else {
      localQpInfo->ece_supported = 0;
      if (ncclParamEnableFaultTolerance()) backupQpInfo->ece_supported = 0;
    }
  }
  return ncclSuccess;
}

// The function modifies the QPs on the sender side to RTR and RTS states. It
// uses the remote metadata (remMeta) provided to the function to get the remote
// QPs' information. The remote metadata is expected to be obtained from the
// remote side (receiver) as part of the connection establishment process.
// Note that if ECE is supported, the function sets up the reduced ECE (which
// was delivered from the receiver side) on the QPs before modifying the QPs
// to RTR.
static ncclResult_t ncclIbSenderQpsToRts(ncclIbSendComm* comm, int dev, struct ncclIbConnectionMetadata* remMeta) {
  uint nqps = comm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbSendCommDev* commDev = &comm->devs[localQp->devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[commDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo   = &remMeta->qpInfo[qpIndex];
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remQpInfo->devIndex];

    ncclIbQp* backupQp = &comm->base.backupQps[qpIndex];
    ncclIbSendCommDev* backupCommDev = &comm->backupDevs[backupQp->devIndex];
    ncclIbDev* backupIbDev = &ncclIbDevs[backupCommDev->base.ibDevN];
    struct ncclIbQpInfo *backupRemQpInfo = &remMeta->backupQpInfo[qpIndex];
    struct ncclIbDevInfo *backupRemDevInfo = &remMeta->backupDevs[backupRemQpInfo->devIndex];

    localQp->remDevIdx = remQpInfo->devIndex;
    backupQp->remDevIdx = backupRemQpInfo->devIndex;

    if (remQpInfo->ece_supported) {
      // Set the reduced ECE received from the receiver side
      INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        commDev->base.ibDevN, ibDev->portNum, localQp->qp->qp_num, remQpInfo->ece_supported, remQpInfo->ece.vendor_id, remQpInfo->ece.options, remQpInfo->ece.comp_mask);
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &remQpInfo->ece_supported));
    }

    remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu); // TODO: This is bad practice!
    NCCLCHECK(ncclIbRtrQp(localQp->qp, &commDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remMeta->tc, remMeta->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));

    memcpy(&comm->base.qps[qpIndex].gidInfo, &commDev->base.gidInfo, sizeof(struct ncclIbGidInfo));
    comm->base.qps[qpIndex].dest_qp_num = remQpInfo->qpn;
    memcpy(&comm->base.qps[qpIndex].info, remDevInfo, sizeof(struct ncclIbDevInfo));
    comm->base.qps[qpIndex].ece_supported = remQpInfo->ece_supported;
    comm->base.qps[qpIndex].ece = remQpInfo->ece;
    comm->base.qps[qpIndex].tc = remMeta.tc;
    comm->base.qps[qpIndex].sl = remMeta.sl;

    // Assign per-QP backup remDev
    if (backupRemQpInfo->ece_supported) {
      INFO(NCCL_NET, "NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        backupCommDev->base.ibDevN, backupIbDev->portNum, backupQp->qp->qp_num, backupRemQpInfo->ece_supported, backupRemQpInfo->ece.vendor_id, backupRemQpInfo->ece.options, backupRemQpInfo->ece.comp_mask);
      NCCLCHECK(wrap_ibv_set_ece(backupQp->qp, &backupRemQpInfo->ece, &backupRemQpInfo->ece_supported));
    }

    ncclIbDev *backupIbDev = ncclIbDevs + backupCommDev->base.ibDevN;
    backupRemDevInfo->mtu = std::min(backupRemDevInfo->mtu, backupIbDev->portAttr.active_mtu);
    NCCLCHECKGOTO(ncclIbRtrQp(qp, &backupCommDev->base.gidInfo, backupRemQpInfo->qpn, backupRemDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);
  }
  return ncclSuccess;
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

  // Create QPs on the sender side
  NCCLCHECKGOTO(ncclIbSenderQpsCreate(comm, &meta), ret, fail);

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

    ncclIbDevInfo *backupDevInfo = NULL;
    if (ncclParamEnableFaultTolerance()) {
      backupDevInfo = meta.backupDevs + i;
      backupDevInfo->ib_port = backupIbDev->portNum;
      backupDevInfo->mtu = backupIbDev->portAttr.active_mtu;
      backupDevInfo->lid = backupIbDev->portAttr.lid;
    } 

    // Prepare GIN Put Signal scratchpad (for RDMA Atomic result)
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr, commDev->base.pd, &comm->putSignalScratchpad, sizeof(comm->putSignalScratchpad), IBV_ACCESS_LOCAL_WRITE), ret, fail);

    // Prepare my CTS FIFO
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->ctsFifoMr, commDev->base.pd, comm->ctsFifo, sizeof(comm->ctsFifo), IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->rkey = commDev->ctsFifoMr->rkey;

    if (ncclParamEnableFaultTolerance()) {
      // Prepare CTS FIFO with backup device
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupCommDev->ctsFifoMr, backupCommDev->base.pd, comm->ctsFifo, sizeof(comm->ctsFifo), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      backupDevInfo->rkey = backupCommDev->ctsFifoMr->rkey;

      // Prepare syncFifo
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->syncFifoMr, commDev->base.pd, comm->syncFifo, sizeof(comm->syncFifo), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      devInfo->syncFifoRkey = commDev->syncFifoMr->rkey;

      // Prepare backup syncFifo
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupCommDev->syncFifoMr, backupCommDev->base.pd, comm->syncFifo, sizeof(comm->syncFifo), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
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
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d subnet-prefix %lu  FLID %d ctsFifoRkey=0x%x ctsFifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
               dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
               (uint64_t)devInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&devInfo->gid), commDev->ctsFifoMr->rkey, commDev->ctsFifoMr->lkey);
        } else { // RoCE
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX) ctsFifoRkey=0x%x ctsFifoLkey=0x%x",
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu,
               (int64_t)commDev->base.gidInfo.localGidIndex,
               (uint64_t)devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, commDev->ctsFifoMr->rkey, commDev->ctsFifoMr->lkey);
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
          if (meta.backupQpInfo[q].ece_supported) {
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
  meta.addr = (uint64_t)comm->ctsFifo;
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

  // Copy remDevInfo for things like remGidInfo, remCmplsRecordsFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;

    if (ncclParamEnableFaultTolerance()) {
      comm->base.backupRemDevs[i] = remMeta.backupDevs[i];
      comm->base.backupRemDevs[i].remoteGid.global.interface_id = comm->base.backupRemDevs[i].gid.global.interface_id;
      comm->base.backupRemDevs[i].remoteGid.global.subnet_prefix = comm->base.backupRemDevs[i].gid.global.subnet_prefix;
    }
  }

  // Retain remote completion records info and prepare RDMA ops
  comm->remCmplsRecords.addr = remMeta.addr;
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->remCmplsRecords.rkeys[i] = remMeta.devs[i].rkey;
    if (ncclParamEnableFaultTolerance()) comm->remCmplsRecords.backupRkeys[i] = remMeta.backupDevs[i].rkey;
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
    ncclIbSendCommDev* commDev = comm->devs + i;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd, &comm->remCmplsRecords.elems, sizeof(comm->remCmplsRecords.elems), IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;

    if (ncclParamEnableFaultTolerance()) {
      ncclIbSendCommDev *backupCommDev = comm->backupDevs + i;
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupCommDev->cmplsRecordsMr, backupCommDev->base.pd, &comm->remCmplsRecords.elems, sizeof(comm->remCmplsRecords.elems), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      backupCommDev->sge.lkey = backupCommDev->cmplsRecordsMr->lkey;
    }
  }
  comm->base.nRemDevs = remMeta.ndevs;

  NCCLCHECKGOTO(ncclIbSenderQpsToRts(comm, dev, &remMeta), ret, fail);

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, comm->base.nRemDevs);

  comm->base.ready = 1;
  comm->base.splitDataOnQps = ncclParamIbSplitDataOnQps();
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

// The function creates and modifies QPs to RTS state on the receiver side
// using remote information from the sender side (remMeta). It also populates
// the remote metadata structure, provided to the function (remMeta), with the
// QPs' information so that data structure could be delivered to the remote
// side (sender) as part of the connection establishment process.
static ncclResult_t ncclIbReceiverQpsCreateToRts(ncclIbRecvComm* rComm, struct ncclIbConnectionMetadata* remMeta, struct ncclIbConnectionMetadata* meta) {
  uint nqps = rComm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % rComm->base.vProps.ndevs;
    ncclIbRecvCommDev* rCommDev = &rComm->devs[devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[rCommDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];
    int remDevIndex = remQpInfo->devIndex;
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remDevIndex];
    ncclIbQp* localQp = &rComm->base.qps[qpIndex];

    ncclIbRecvCommDev* backupRCommDev = NULL;
    ncclIbDev* backupIbDev = NULL;
    ncclIbQpInfo* backupRemQpInfo = NULL;
    ncclIbQpInfo* backupLocalQpInfo = NULL;
    ncclIbDevInfo* backupRemDevInfo = NULL;
    ncclIbQp* backupLocalQp = NULL;
    int backupRemDevIndex = -1;
    if (ncclParamEnableFaultTolerance()) {
      backupRCommDev = &rComm->backupDevs[devIndex];
      backupIbDev = &ncclIbDevs[backupRCommDev->base.ibDevN];
      backupRemQpInfo = &remMeta->backupQpInfo[qpIndex];
      backupLocalQpInfo = &meta->backupQpInfo[qpIndex];
      backupRemDevIndex = backupRemQpInfo->devIndex;
      backupRemDevInfo = &remMeta->backupDevs[backupRemDevIndex];
      backupLocalQp = &rComm->base.backupQps[qpIndex];
    }

    localQp->remDevIdx = remDevIndex;
    localQp->devIndex = devIndex;

    if (ncclParamEnableFaultTolerance()) {
      backupLocalQp->remDevIdx = backupRemDevIndex;
      backupLocalQp->devIndex = devIndex;
    }

    NCCLCHECK(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC, &rComm->base.stats, localQp, false));

    localQpInfo->qpn      = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    if (ncclParamEnableFaultTolerance()) {
      NCCLCHECK(ncclIbCreateQp(backupIbDev->portNum, &backupRCommDev->base, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC, &rComm->base.backupStats, backupLocalQp, true));

      backupLocalQpInfo->qpn      = backupLocalQp->qp->qp_num;
      backupLocalQpInfo->devIndex = backupLocalQp->devIndex;
    }

    // Set ECE (enhanced connection establishment) on before RTR
    if (remQpInfo->ece_supported) {
      // coverity[copy_paste_error]
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &localQpInfo->ece_supported));
    } else {
      localQpInfo->ece_supported = 0;
    }

    // Reduce the local MTU to match the remote MTU if needed
    ibDev->portAttr.active_mtu = std::min(ibDev->portAttr.active_mtu, remDevInfo->mtu);

    NCCLCHECK(ncclIbRtrQp(localQp->qp, &rCommDev->base.gidInfo, remQpInfo->qpn, remDevInfo, true, remMeta->tc, remMeta->sl));
    NCCLCHECK(ncclIbRtsQp(localQp->qp));

    // Query the reduced ECE by the device and storing it in the local QP info
    // to return it to the requestor (sender).
    if (remQpInfo->ece_supported && localQpInfo->ece_supported) {
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQpInfo->ece_supported));
    }

    if (ncclParamEnableFaultTolerance()) {
      // Set the backup ece (enhanced connection establishment) on this QP before RTR
      if (backupRemQpInfo->ece_supported) {
        // coverity[copy_paste_error]
        NCCLCHECKGOTO(wrap_ibv_set_ece(backupQp->qp, &backupRemQpInfo->ece, &backupLocalQpInfo->ece_supported), ret, fail);
      }
      else {
        backupLocalQpInfo->ece_supported = 0;
      }
      // Reduce the local MTU to match the remote MTU if needed
      backupIbDev->portAttr.active_mtu = std::min(backupIbDev->portAttr.active_mtu, backupRemDevInfo->mtu);

      NCCLCHECK(ncclIbRtrQp(backupQp->qp, &backupRCommDev->base.gidInfo, backupRemQpInfo->qpn, backupRemDevInfo, true, remMeta.tc, remMeta.sl));
      NCCLCHECK(ncclIbRtsQp(backupQp->qp));

      // Query the reduced ECE by the device and storing it in the local QP info
      // to return it to the requestor (sender).
      if (backupRemQpInfo->ece_supported && backupLocalQpInfo->ece_supported) {
        NCCLCHECK(wrap_ibv_query_ece(backupQp->qp, &backupLocalQpInfo->ece, &backupLocalQpInfo->ece_supported));
      }

      memcpy(&localQp->gidInfo, &rCommDev->base.gidInfo, sizeof(struct ncclIbGidInfo));
      localQp->dest_qp_num = remMeta.qpInfo[q].qpn;
      memcpy(&localQp->info, remDevInfo, sizeof(struct ncclIbDevInfo));
      localQp->ece_supported = remMeta.qpInfo[q].ece_supported;
      localQp->ece = remMeta.qpInfo[q].ece;
      localQp->tc = remMeta.tc;
      localQp->sl = remMeta.sl;
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = lComm->stage;
  if (stage == NULL) {
    NCCLCHECK(ncclCalloc(&lComm->stage, 1));
    stage = lComm->stage;
  }
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

  struct ncclIbDev *backupIbDev;
  int backupIbDevN;
  struct ncclIbRecvCommDev *backupRCommDev;

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

  // Copy remGidInfo, remCtsFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;

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

  NCCLCHECKGOTO(ncclIbReceiverQpsCreateToRts(rComm, &remMeta, &meta), ret, fail);

  rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;
  if (ncclParamEnableFaultTolerance()) rComm->backupFlushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev ^ 1) == ncclSuccess) 
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  // Retain remote CTS FIFO info and prepare my RDMA ops
  rComm->remCtsFifo.addr = remMeta.addr;
  if (ncclParamEnableFaultTolerance()) rComm->remSyncFifo.addr = remMeta.syncFifoAddr;
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->remCtsFifo.rkeys[i] = remMeta.devs[i].rkey;
    if (ncclParamEnableFaultTolerance()) rComm->remCtsFifo.backupRkeys[i] = remMeta.backupDevs[i].rkey;
  }
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    if (ncclParamEnableFaultTolerance()) backupRCommDev = rComm->backupDevs + i;

    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->ctsFifoMr, rCommDev->base.pd, &rComm->remCtsFifo.elems, sizeof(rComm->remCtsFifo.elems), IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    rCommDev->sge.lkey = rCommDev->ctsFifoMr->lkey;

    // Prepare completion records
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->cmplsRecordsMr, rCommDev->base.pd, &rComm->cmplsRecords, sizeof(rComm->cmplsRecords), IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;

    if (ncclParamEnableFaultTolerance()) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupRCommDev->ctsFifoMr, backupRCommDev->base.pd, &rComm->remCtsFifo.elems, sizeof(rComm->remCtsFifo.elems), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      backupRCommDev->sge.lkey = backupRCommDev->ctsFifoMr->lkey;

      // Prepare completion records
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&backupRCommDev->cmplsRecordsMr, backupRCommDev->base.pd, &rComm->cmplsRecords, sizeof(rComm->cmplsRecords), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      meta.backupDevs[i].rkey = backupRCommDev->cmplsRecordsMr->rkey;

      // Retain remote sync FIFO info and prepare my RDMA ops
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->syncFifoMr, rCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(rComm->remSyncFifo.elems), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ), ret, fail);
      rCommDev->syncFifoSge.lkey = rCommDev->syncFifoMr->lkey;

      // backup Retain remote sync fifo info and prepare my RDMA ops
      NCCLCHECK(wrap_ibv_reg_mr(&backupRCommDev->syncFifoMr, backupRCommDev->base.pd, &rComm->remSyncFifo.elems, sizeof(rComm->remSyncFifo.elems), IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ));
      backupRCommDev->syncFifoSge.lkey = backupRCommDev->syncFifoMr->lkey;
    }
  }
  if (ncclParamIbUseInline()) rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = ncclIbDevs + rCommDev->base.ibDevN;

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
    if (ncclParamEnableFaultTolerance() && rComm->backupFlushEnabled) {
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
  }
  meta.addr = (uint64_t)rComm->cmplsRecords;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

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

  rComm->base.splitDataOnQps = ncclParamIbSplitDataOnQps();

  *recvComm = rComm;
exit:
  /* reset lComm stage */
  if (stage->buffer) free(stage->buffer);
  free(stage);
  lComm->stage = NULL;
  return ret;
fail:
  free(rComm);
  goto exit;
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
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (ncclParamEnableFaultTolerance() && commDev->syncFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->syncFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
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
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (ncclParamEnableFaultTolerance() && commDev->syncFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->syncFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
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
        (ncclIbDestroyBase(&backupCommDev->base, true));
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
