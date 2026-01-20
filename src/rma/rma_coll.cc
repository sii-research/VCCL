/**
 * CC operations implemented in RMA proxy and CE.
 */
#include <assert.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"

ncclResult_t ncclLaunchRmaColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  return ncclSuccess;
}

ncclResult_t scheduleRmaCollTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  return ncclSuccess;
}
