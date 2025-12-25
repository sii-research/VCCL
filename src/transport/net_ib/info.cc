#include "common.h"
#include "compiler.h"

ncclResult_t saveChannelToQp(void *netSendComm, int channel_id) {
  struct ncclIbSendComm *comm = (struct ncclIbSendComm *)netSendComm;
  for (int q = 0; q < comm->base.nqps; q++) {
    comm->base.qps[q].channel_id = channel_id;
    comm->base.backupQps[q].channel_id = channel_id;
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