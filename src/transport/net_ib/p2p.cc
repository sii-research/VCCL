/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p.h"
#include "common.h"
#include "compiler.h"

NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
extern int64_t ncclParamEnableFaultTolerance();

const char* ncclIbReqTypeStr[] = { "Unused", "Send", "Recv", "Flush", "IPut" };

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
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

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, bool if_backup) {
  req->events[devIndex]++;
  struct ncclIbNetCommDevBase *base = ncclIbGetNetCommDevBase(req->base, devIndex, if_backup);
  if (if_backup) {
    req->backupDevBases[devIndex] = base;
  }
  else {
    req->devBases[devIndex] = base;
  }
}

// count the number of send wrs and remain data sizes in wrs
thread_local int sendWrCounter[NCCL_IB_MAX_DEVS_PER_NIC] = {0};
thread_local int remainWrDataSize[NCCL_IB_MAX_DEVS_PER_NIC] = {0};

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
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

  // When nreqs==1, the Immediate Data carries the size of the send request.
  // In case of a multi-send (nreqs>1), the Immediate Data is ignored by the
  // receiver, as the size of the send request is written by the sender side
  // directly to the remote completion records array. Therefore, always
  // assigning the Immediate Data with the size, does not harm, and when it's
  // not required - it's ignored by the receiver side.
  uint32_t immData = reqs[0]->send.size;
  if (nreqs > 1) {
    int* sizes = comm->remCmplsRecords.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
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
      lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = htobe32(immData);
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);

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

  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
    int devIndex = qp->devIndex;

    // check if qp is available
    bool if_backup = false;
    if (ncclParamEnableFaultTolerance() && comm->devs[devIndex].base.warn.is_warn == true) {
      NCCLCHECK(ncclIbCommBaseGetBackupQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
      devIndex = qp->devIndex;
      if_backup = true;
    }

    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex);

      // update sendWrCounter
      sendWrCounter[devIndex]++;
      // Select proper rkey (needed even for 0-size send)
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
      // Populating the correct gather information based on the device and
      // slot used.
      // Note that the lkey is already correct from the initialization phase.
      lastWr->sg_list = if_backup ? &(comm->backupDevs[devIndex].sge) : &(comm->devs[devIndex].sge);
      lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);
      lastWr->sg_list[0].length = nreqs*sizeof(int);
      // Populate the correct RKey based on the device used
      lastWr->wr.rdma.rkey = if_backup ? comm->remCmplsRecords.backupRkeys[devIndex] : comm->remCmplsRecords.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
#ifdef NCCL_ENABLE_NET_PROFILING
    // QP profiling loop
    for (int r=0; r<nreqs; r++) {
      // Store the qpIndex for this request
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

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  std::atomic_thread_fence(std::memory_order_seq_cst); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // choose normal qp or backup qp according to the backup flag
    // we ensure use both normal or backup qp in dual ports
    if (slots[r].if_backup != comm->devs[0].base.warn.is_warn) {
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
    int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    int qpIndex = -1;
    ncclIbQp* qp = NULL;
    for (int i = 0; i < nqps; i++) {
      NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));

      bool if_backup = false;
      if (ncclParamEnableFaultTolerance() && comm->devs[qp->devIndex].base.warn.is_warn == true) {
        NCCLCHECK(ncclIbCommBaseGetBackupQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
        if_backup = true;
      }

      // Add event
      ncclIbAddEvent(req, qp->devIndex, if_backup);

      int devIndex = qp->devIndex;
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
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      bool if_backup = false;
      if (ncclParamEnableFaultTolerance() && comm->devs[qp->devIndex].base.warn.is_warn == true) {
        if_backup = true;
      }

      if (!if_backup) req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
      else req->send.lkeys[i] = mhandleWrapper->mrs[i + NCCL_IB_MAX_DEVS_PER_NIC]->lkey;
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
    comm->base.fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}


// change the fifo head
ncclResult_t ncclIbChangeFifoHead(void *sendComm, uint64_t _FifoHead) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)sendComm;
  comm->base.fifoHead = _FifoHead;
  return ncclSuccess;
}

ncclResult_t ncclIbCheckSubSyncFifo(void *sendComm, bool &if_rollback) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)sendComm;
  volatile struct ncclIbSyncFifo *slots;

  int slot = (comm->base.syncFifoHead) % MAX_REQUESTS;
  slots = comm->syncFifo;
  uint64_t idx = comm->base.syncFifoHead + 1;
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

  int slot = (comm->base.syncFifoHead) % MAX_REQUESTS;
  slots = comm->syncFifo;
  uint64_t idx = comm->base.syncFifoHead + 1;

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

  comm->base.syncFifoHead++;
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
  ncclIbQp *backupCtsQp = NULL;
  NCCLCHECK(ncclIbRecvCommGetBackupQpForCts(comm, comm->base.fifoHead, &backupCtsQp));

  comm->base.fifoHead += 1000;

  localElem->recvFifoTail = comm->base.fifoHead;
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

  int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
  req->recv.sizes = comm->cmplsRecords[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];

  ncclIbQp* ctsQp = NULL;;
  NCCLCHECK(ncclIbRecvCommGetQpForCts(comm, comm->base.fifoHead, &ctsQp));

  ncclIbQp* backupCtsQp = NULL;
  if (ncclParamEnableFaultTolerance()) {
    NCCLCHECK(ncclIbRecvCommGetBackupQpForCts(comm, comm->base.fifoHead, &backupCtsQp));
  }

  bool if_backup = false;
  if (ncclParamEnableFaultTolerance() && comm->devs[ctsQp->devIndex].base.warn.is_warn == true) {
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
    localElem[i].idx = comm->base.fifoHead+1;
  }
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct rkey
  if (!if_backup) {
    wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  }
  else {
    wr.wr.rdma.rkey = comm->base.backupRemDevs[backupCtsQp->remDevIdx].rkey;
  }

  // Populating the correct gather information based on the device and user
  // provided information
  if (!if_backup) wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  else wr.sg_list = &(comm->backupDevs[backupCtsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;
  wr.sg_list[0].length = n*sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

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
  // slot == devIndex - When writing to CTS FIFO slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each CTS posting QP gets drained
  if (!if_backup) {
    if (((comm->base.vProps.ndevs == 1) && (slot == 0)) ||
        ((comm->base.vProps.ndevs > 1) && (slot == 0 || slot == 1))) {
      wr.send_flags |= IBV_SEND_SIGNALED;
      wr.wr_id = req - comm->base.reqs;
      ncclIbAddEvent(req, ctsQp->devIndex, if_backup);

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
      ncclIbAddEvent(req, backupCtsQp->devIndex, if_backup);

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
  comm->base.fifoHead++;

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

  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  int qpIndex = -1;
  ncclIbQp* qp = NULL;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbCommBaseGetQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
    bool if_backup = false;
    // check if qp is available
    if (ncclParamEnableFaultTolerance() && comm->devs[qp->devIndex].base.warn.is_warn == true) {
      if_backup = true;
      NCCLCHECK(ncclIbCommBaseGetBackupQpForRequest(&comm->base, comm->base.fifoHead, i, &qp, &qpIndex));
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
      req->pInfo[r].qpIndex[nEventHandles] = qpIndex;
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

    ncclIbAddEvent(req, i, if_backup);
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
    // lComm->base.qpIndex = 0;
    // In nccl2.29.2, qpIndex is got from fifoHead%nqpsPerRequest, so no need to reset qpIndex here
  }
  else {
    struct ncclIbRecvComm *rComm = (struct ncclIbRecvComm *)comm;
    // rComm->base.qpIndex = 0;
    // In nccl2.29.2, qpIndex is got from fifoHead%nqpsPerRequest, so no need to reset qpIndex here
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

static inline ncclResult_t ncclIbRequestRetrieveAsIndex(ncclIbRequest* reqs, uint32_t reqIndex, ncclIbRequest** req) {
  if (reqIndex < 0 || reqIndex >= NET_IB_MAX_REQUESTS) {
    WARN("NET/IB: %s: Invalid request index %d. Not in the range [%d, %d). Cannot retrieve request.", __func__, reqIndex, 0, NET_IB_MAX_REQUESTS);
    return ncclInternalError;
  }
  *req = &reqs[reqIndex];
  return ncclSuccess;
}

static inline bool ncclIbRequestIsComplete(struct ncclIbRequest *request) {
  return (request->events[0] == 0 && request->events[1] == 0 && request->events[2] == 0 && request->events[3] == 0);
}

static inline ncclResult_t ncclIbRequestComplete(struct ncclIbRequest* r, int* done, int* sizes) {
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

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t ncclIbLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(commBase, devIndex, false);
  char localGidString[INET6_ADDRSTRLEN] = "";
  char remoteGidString[INET6_ADDRSTRLEN] = "";
  const char* localGidStr = NULL, *remoteGidStr = NULL;
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
    remoteGidStr = ibvGetGidStr(&commBase->remDevs[devIndex].remoteGid, remoteGidString, sizeof(remoteGidString));
  }

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  r->devBases[devIndex]->warn.is_warn = true;
  if (r->devBases[devIndex ^ 1] != NULL) r->devBases[devIndex ^ 1]->warn.is_warn = true;
  r->devBases[devIndex]->warn.line = addr;
  r->devBases[devIndex]->warn.status = wc->status;
  r->devBases[devIndex]->warn.opcode = wc->opcode;
  r->devBases[devIndex]->warn.len = wc->byte_len;
  r->devBases[devIndex]->warn.error = wc->vendor_err;
  r->devBases[devIndex]->warn.type = reqTypeStr[r->type];
  if (r->devBases[devIndex]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    r->devBases[devIndex]->warn.localGidstr = localGidStr ? " localGid " : "";
    r->devBases[devIndex]->warn.localGidstring = localGidString;
    r->devBases[devIndex]->warn.remoteGidstr = remoteGidStr ? " remoteGids" : "";
    r->devBases[devIndex]->warn.remoteGidstring = remoteGidString;
  }
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u %s%s%s%s hca %s",
      sockStr, ibvWcStatusStr(wc->status), wc->status,
      ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err,
      localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
  return ncclSuccess;
}

static inline ncclResult_t ncclIbCompletionEventProcess(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);

  struct ncclIbRequest* req = NULL;
  NCCLCHECK(ncclIbRequestRetrieveAsIndex(commBase->reqs, wc->wr_id & 0xff, &req));

  #ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN+1];
  TRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d,%d,%d}, devIndex=%d",
    ncclSocketToString(&addr, line), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], req->events[2], req->events[3], devIndex);
  #endif
  if (req && req->type == NCCL_NET_IB_REQ_SEND) {
    // update sendWrCounter
    sendWrCounter[i] -= req->nreqs;

    for (int j = 0; j < req->nreqs; j++) {
      struct ncclIbRequest* sendReq = NULL;
      NCCLCHECK(ncclIbRequestRetrieveAsIndex(commBase->reqs, (wc->wr_id >> (j*8)) & 0xff, &sendReq));
      if ((sendReq->events[devIndex] <= 0)) {
        WARN("NET/IB: sendReq(%p)->events={%d,%d,%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], sendReq->events[2], sendReq->events[3], devIndex, j);
        ret = ncclInternalError;
        goto ret;
      }
      sendReq->events[devIndex]--;
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
        ret =  ncclInternalError;
        goto ret;
      }
      if (req->nreqs == 1) {
        req->recv.sizes[0] = be32toh(wc->imm_data);
      }
    }
    req->events[devIndex]--;
#ifdef NCCL_ENABLE_NET_PROFILING
    // Stop Qp event for workFifo
    for (int j = 0; j < req->nreqs; j++) {
      int qpIndex = getReqQpIndex(req, j, wc->qp_num);
      NCCLCHECK(ncclProfilerFunction(&req->pInfo[j].qpEventHandles[qpIndex], ncclProfilerNetEventStop, NULL, 0, NULL));
    }
#endif
  }
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;
  int totalWrDone = 0;
  int wrDone = 0;
  int backupWrDone = 0;
  struct ibv_wc wcs[8];
  do {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (ncclIbRequestIsComplete(r)) {
      NCCLCHECK(ncclIbRequestComplete(r, done, sizes));
      return ncclSuccess;
    }

    totalWrDone = 0;
    for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      // If we expect any completions from this device's CQ
      if (r->events[i] == 0) {
        continue;
      }
      TIME_START(3);
      if (r->devBases[i] != NULL && !r->devBases[i]->warn.is_warn) {
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
          ncclIbLogCompletionWithError(r->base, wc, i);
          ret = ncclRemoteError;
          goto ret;
        }
        NCCLCHECK(ncclIbCompletionEventProcess(r->base, wc, i));
      }
      // Once the IB fatal event is reported in the async thread, we want to propagate this error
      // to communicator and prevent further polling to reduce error pollution.
      // NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
    }
  } while (totalWrDone > 0);

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

  // If no (more) CQEs found on any device, return and come back later
  if (totalWrDone == 0) {
    ret = ncclSuccess;
    goto ret;
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
