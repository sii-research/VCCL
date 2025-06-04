#ifndef TIMER_LOG_H
#define TIMER_LOG_H

#include <mutex>
#include <stdlib.h>
#include <stdio.h>
#include <timer.h>
#include <sys/types.h>
#include <string.h>
// #include <cstdint>
#include <pthread.h>
#include <signal.h>
#include "socket.h"
#include "nccl.h"
#include "core.h"
#include <deque>
#include <queue>
#include <stack>

enum timer_log_type{
  NCCL_LOG_NOT_USE = 0,
  NCCL_LOG_TELEMETRY = 1,
  NCCL_LOG_HANG = 2
};

struct timer_log{
  int rank;
  int channel_id;
  uint8_t func;
  unsigned long long ncclFuncTimes;
  uint8_t srcIp[4];
  uint8_t dscIp[4];
  struct timespec send_start;
  struct timespec send_end;  
  int loged_start;
  int loged_end;
  unsigned long long diff;
  int size = 0;
  double rate;
  std::string NetworkCardName;
  int peerRank;
  uint64_t groupHash;
  int sendWrCounter;
  int devIndex;
  int remainWrDataSize;
};

// define the size of windowsSize
const int maxWindowSize = ncclGetEnv("TELEMETRY_WINDOWSIZE") ? 
                          atoi(ncclGetEnv("TELEMETRY_WINDOWSIZE")) : 
                          50;

enum linkStatus{
  LINK_STATUS_UNUSED,
  LINK_STATUS_USED,
  LINK_STATUS_WAIT,
  LINK_STATUS_SUCCESS,
  LINK_STATUS_WRONG,
  LINK_STATUS_WRONG_WAIT
};

struct linkStatusTest{
  int status;
  struct ibv_qp *linkPingQp;
  struct timespec send_start;
  struct timespec send_end;
  int events;
};

extern const int nccl_telemetry_enable;
extern const char* nccl_telemetry_log_path;

//INFO(NCCL_ENV, "NCCL_TELEMETRY_ENABLE is set to %d", nccl_telemetry_enable);
//INFO(NCCL_ENV, "NCCL_TELEMETRY_LOG_PATH is set to %s", nccl_telemetry_log_path);



#define TIMER_LOG_ENTRY           nccl_telemetry_enable
#define TIMER_LOG_NCCL_HANG       nccl_telemetry_enable
#define TIMER_LOG_NCCL_TELEMETRY  nccl_telemetry_enable


#define TIMER_LOG_QUEUE_READ -1
#define TIMER_LOG_QUEUE_WRITE 1

#define SOCK_PATH "/tmp/unix_sock"
void* timerLogService(void *args);
void printLogInfo(struct timer_log log);


#define TIMER_LOG_MAX_LEN 50010
struct timer_log_queue{
  pthread_t thread;
  pthread_mutex_t lock;
  std::mutex telemetryStateLock;
  volatile int state;
  volatile int stop;
  std::deque<timer_log> log;
  std::queue<timer_log> slideWindow[4];     // different port
  volatile unsigned long long windowDataSizes[4];
  volatile unsigned long long sendEndTime[4];  // count the timestamp of the last log in slideWindow
  volatile timer_log* lastLog[4];             // the last log of different devIndex in Log, for push operation when log is full
  volatile bool collect;
  volatile int live = -1;

  void push(struct timer_log& _log){
    if (log.size() >= TIMER_LOG_MAX_LEN) {
      if (lastLog[_log.devIndex]!= NULL && lastLog[_log.devIndex]->ncclFuncTimes == _log.ncclFuncTimes) {
        // merge the new log into before one
        lastLog[_log.devIndex]->size += _log.size;
        lastLog[_log.devIndex]->diff = _log.diff;
        lastLog[_log.devIndex]->func = _log.func;
      }
      else {
        // merge the first two log that have same ncclFuncTimes
        std::stack<timer_log> stk;
        while(!log.empty()){
          timer_log frontLog = log.front();
          log.pop_front();
          if (!stk.empty() && stk.top().ncclFuncTimes == frontLog.ncclFuncTimes && stk.top().devIndex == frontLog.devIndex) {
            stk.top().size += frontLog.size;
            stk.top().diff = frontLog.diff;
            break;
          }
          stk.push(frontLog);
        }
        while(!stk.empty()) {
          timer_log topLog = stk.top();
          log.push_front(topLog);
          stk.pop();
        }
        log.push_back(_log);
        lastLog[_log.devIndex] = &log.back();
      }
    }
    else {
      log.push_back(_log);
      lastLog[_log.devIndex] = &log.back();
    }
    return;
  }
  struct timer_log pop(){
    if (log.empty()) {
      struct timer_log res;
      memset((void *)&res, 0, sizeof(res));
      return res;
    }
    timer_log popLog = log.front();
    // judge if the pop one is the last log of its devIndesx
    if (lastLog[popLog.devIndex] == &log.front())
    {
      lastLog[popLog.devIndex] = NULL;
    }

    log.pop_front(); 
    return popLog;
  }
  bool empty() {
    while (!log.empty()) {
      log.pop_back();
    }
    return true;
  }

  void pushSlideWindow(struct timer_log& _log, int devIndex) {
    if (!slideWindow[devIndex].empty() && 
        (slideWindow[devIndex].back().ncclFuncTimes != _log.ncclFuncTimes)) {
      emptySlideWindow(devIndex);
      lastLog[0] = NULL;
      lastLog[1] = NULL;
      lastLog[2] = NULL;
      lastLog[3] = NULL;
      windowDataSizes[devIndex] = 0;
    }
    if (slideWindow[devIndex].size() >= maxWindowSize) {
      timer_log frontLog = slideWindow[devIndex].front();
      slideWindow[devIndex].pop();
      windowDataSizes[devIndex] -= frontLog.size;
    }
    slideWindow[devIndex].push(_log);
    sendEndTime[devIndex] = _log.diff;
    windowDataSizes[devIndex] += _log.size;
    return;
  }

  void popSlideWindow(int devIndex) {
    if (!slideWindow[devIndex].empty()) {
      slideWindow[devIndex].pop();
    }
    return;
  }

  void emptySlideWindow(int devIndex) {
    while (!slideWindow[devIndex].empty()) {
      slideWindow[devIndex].pop();
    }
    return;
  }

  int getBandWidths(int devIndex) {
    if (slideWindow[devIndex].size() <= 1) {
      return 0;
    }

    // Gbps
    unsigned long long sendTime = sendEndTime[devIndex] - slideWindow[devIndex].front().diff;
    unsigned long long sendDataSizes = windowDataSizes[devIndex] - slideWindow[devIndex].front().size;
    // 0.93 = 1e9 / 1024 * 1024 * 1024
    return sendDataSizes * 0.93 * 8 / sendTime;
  }

  void init(){
    std::lock_guard<std::mutex> stateLock(telemetryStateLock);
    if(live == -1){
      state = 0;
      stop = 0;
      windowDataSizes[0] = 0;
      windowDataSizes[1] = 0;
      windowDataSizes[2] = 0;
      windowDataSizes[3] = 0;
      lastLog[0] = NULL;
      lastLog[1] = NULL;
      lastLog[2] = NULL;
      lastLog[3] = NULL;
      collect = 0;
      live = 1;
      pthread_mutex_init(&lock, NULL);
      pthread_create(&thread, NULL, timerLogService, NULL);
    }
  }
  bool setState(int to){
    // return 1;

    if(state != 0) return 0;
    if(pthread_mutex_trylock(&lock) == 0){
      if(state == 0) state = to;
      pthread_mutex_unlock(&lock);
      return state == to;
    }
    return 0;
  } 
  void freeState(){
    // pthread_mutex_lock(&lock);
    state = 0;
    // pthread_mutex_unlock(&lock);
  }
  void destroy(){
    std::lock_guard<std::mutex> stateLock(telemetryStateLock);
    if(live == 1){
      stop = 1;
      pthread_join(thread, nullptr);
      live = 0;
    }
  }
};

#ifdef NET_IB_CC
struct timer_log_queue global_timer_log;
#else
extern struct timer_log_queue global_timer_log;
#endif



/*diff net_ib.cc:
 *#define NET_IB_CC
 *#include "timer_log.h"
 *proxy.cc:
 *pthread_create(&global_timer_log.thread, NULL, timerLogService, NULL);
 */

#endif
