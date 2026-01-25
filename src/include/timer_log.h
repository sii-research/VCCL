// #ifndef TIMER_LOG_H
// #define TIMER_LOG_H

// #include <mutex>
// #include <stdlib.h>
// #include <stdio.h>
// #include <timer.h>
// #include <sys/types.h>
// #include <string.h>
// // #include <cstdint>
// #include <pthread.h>
// #include <signal.h>
// #include "socket.h"
// #include "nccl.h"
// #include "core.h"
// #include <deque>
// #include <queue>
// #include <stack>
// #include <unordered_map>
// #include <vector>
// #include <atomic>

// enum timer_log_type{
//   NCCL_LOG_NOT_USE = 0,
//   NCCL_LOG_TELEMETRY = 1,
//   NCCL_LOG_HANG = 2
// };

// struct timer_log{
//   int rank;
//   int channel_id;
//   uint8_t func;
//   unsigned long long ncclFuncTimes;
//   uint8_t srcIp[4];
//   uint8_t dscIp[4];
//   struct timespec send_start;
//   struct timespec send_end;  
//   int loged_start;
//   int loged_end;
//   unsigned long long diff;
//   int size = 0;
//   double rate;
//   std::string NetworkCardName;
//   int peerRank;
//   uint64_t groupHash;
//   int sendWrCounter;
//   int devIndex;
//   int remainWrDataSize;
//   int bandwidth;
// };

// // define the size of windowsSize
// const int maxWindowSize = ncclGetEnv("TELEMETRY_WINDOWSIZE") ? 
//                           atoi(ncclGetEnv("TELEMETRY_WINDOWSIZE")) : 
//                           50;

// enum linkStatus{
//   LINK_STATUS_UNUSED,
//   LINK_STATUS_USED,
//   LINK_STATUS_WAIT,
//   LINK_STATUS_SUCCESS,
//   LINK_STATUS_WRONG,
//   LINK_STATUS_WRONG_WAIT
// };

// struct linkStatusTest{
//   int status;
//   struct ibv_qp *linkPingQp;
//   struct timespec send_start;
//   struct timespec send_end;
//   int events;
// };

// extern const int nccl_telemetry_enable;
// extern const char* nccl_telemetry_log_path;

// //INFO(NCCL_ENV, "NCCL_TELEMETRY_ENABLE is set to %d", nccl_telemetry_enable);
// //INFO(NCCL_ENV, "NCCL_TELEMETRY_LOG_PATH is set to %s", nccl_telemetry_log_path);



// #define TIMER_LOG_ENTRY           nccl_telemetry_enable
// #define TIMER_LOG_NCCL_HANG       nccl_telemetry_enable
// #define TIMER_LOG_NCCL_TELEMETRY  nccl_telemetry_enable


// #define TIMER_LOG_QUEUE_READ -1
// #define TIMER_LOG_QUEUE_WRITE 1

// #define SOCK_PATH "/tmp/unix_sock"
// void* timerLogService(void *args);
// void printLogInfo(struct timer_log log);


// #define RING_BUFFER_SIZE 262144
// #define RING_BUFFER_MASK (RING_BUFFER_SIZE - 1)

// template <typename T>
// class LogQueue {
// public:
//   LogQueue() : head_(0), tail_(0) {
//     buffer_.resize(RING_BUFFER_SIZE);
//   }

//   bool enqueue(const T &item) {
//     const size_t curr_tail = tail_.load(std::memory_order_relaxed);
//     const size_t next_tail = (curr_tail + 1) & RING_BUFFER_MASK;

//     if (next_tail == head_.load(std::memory_order_acquire)) {
//       return false;
//     }

//     buffer_[curr_tail] = item;
//     tail_.store(next_tail, std::memory_order_release);
//     return true;
//   }

//   bool dequeue(T &item) {
//     const size_t curr_head = head_.load(std::memory_order_relaxed);

//     if (curr_head == tail_.load(std::memory_order_acquire)) {
//       return false;
//     }

//     item = buffer_[curr_head];
//     head_.store((curr_head + 1) & RING_BUFFER_MASK, std::memory_order_release);
//     return true;
//   }

//   bool empty() const {
//     return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_acquire);
//   }

// private:
//   alignas(64) std::atomic<size_t> head_;
//   alignas(64) std::atomic<size_t> tail_;
//   std::vector<T> buffer_;
// };

// struct RemainLogNode {
//   unsigned long long ncclFuncTimes;
//   timer_log log;
//   RemainLogNode* next;
//   RemainLogNode* prev;
// };

// struct RemainLogList {
//   RemainLogNode* head;
//   RemainLogNode* tail;
//   std::unordered_map<unsigned long long, RemainLogNode*> logMap[4];

//   RemainLogList() {
//     head = new RemainLogNode();
//     head->next = nullptr;
//     head->prev = nullptr;
//     tail = head;
//   }

//   ~RemainLogList() {
//     RemainLogNode* current = head;
//     while (current) {
//       RemainLogNode* temp = current;
//       current = current->next;
//       delete temp;
//     }
//   }
// public:
//   void push(const timer_log& log) {
//     if (logMap[log.devIndex].find(log.ncclFuncTimes) != logMap[log.devIndex].end()) {
//       // Directly merge to existing log
//       RemainLogNode* existingNode = logMap[log.devIndex][log.ncclFuncTimes];
//       existingNode->log.size += log.size;
//       existingNode->log.diff = log.diff;
//       existingNode->prev->next = existingNode->next;
//       if (existingNode->next) {
//         existingNode->next->prev = existingNode->prev;
//       } else {
//         tail = existingNode->prev;
//       }
//       existingNode->next = nullptr;
//       existingNode->prev = tail;
//       tail->next = existingNode;
//       tail = existingNode;
//       return;
//     }
//     RemainLogNode* newNode = new RemainLogNode();
//     newNode->ncclFuncTimes = log.ncclFuncTimes;
//     newNode->log = log;
//     newNode->next = nullptr;
//     newNode->prev = tail;
//     tail->next = newNode;
//     tail = newNode;
//     logMap[log.devIndex][log.ncclFuncTimes] = newNode;
//     return;
//   }

//   bool get_front(timer_log& log) {
//     if (head->next == nullptr) return false;
//     RemainLogNode* temp = head->next;
//     log = temp->log;
//     return true;
//   }

//   bool pop_front() {
//     if (head->next == nullptr) return false;
//     RemainLogNode *temp = head->next;
//     if (temp == tail) {
//       tail = head;
//     }
//     head->next = temp->next;
//     logMap[temp->log.devIndex].erase(temp->ncclFuncTimes);
//     delete temp;
//     return true;
//   }
// };

// struct timer_log_queue{
//   pthread_t thread;
//   pthread_mutex_t lock;
//   std::mutex telemetryStateLock;
//   volatile int state;
//   volatile int stop;
//   LogQueue<timer_log> log;
//   std::queue<timer_log> slideWindow[4];     // different port
//   volatile unsigned long long windowDataSizes[4]; 
//   volatile unsigned long long sendEndTime[4];  // count the timestamp of the last log in slideWindow
//   RemainLogList remainLog;
//   volatile bool collect;
//   volatile int live = -1;
//   int maxNcclFuncTimes = 0;

//   void push(struct timer_log& _log){
//     // First check if there are remained logs to be enqueued first
//     timer_log remainLogItem;
//     while (remainLog.get_front(remainLogItem)) {
//       if (log.enqueue(remainLogItem)) {
//         remainLog.pop_front();
//         continue;
//       } else {
//         break;
//       }
//     }

//     // Second enqueue the new log
//     if (log.enqueue(_log)) {
//       return;
//     }

//     // Third, if the log queue is full, try to merge the new log into last one
//     remainLog.push(_log);
//     return;
//   }

//   struct timer_log pop(){
//     timer_log _log;
//     if (!log.dequeue(_log)) {
//       // return an empty log
//       timer_log emptyLog;
//       memset((void*)&emptyLog, 0, sizeof(timer_log));
//       return emptyLog;
//     }
//     return _log;
//   }

//   struct std::vector<timer_log> pop_all(){
//     std::vector<timer_log> res;
//     timer_log _log;
//     while (log.dequeue(_log)) {
//       res.push_back(_log);
//     }
//     return res;
//   }

//   bool empty() {
//     timer_log _log;
//     while (log.dequeue(_log)) {
//       continue;
//     }
//     return true;
//   }

//   void pushSlideWindow(struct timer_log& _log, int devIndex) {
//     if (!slideWindow[devIndex].empty() && _log.ncclFuncTimes > maxNcclFuncTimes) {
//       int prev_bandwidths = getBandWidths(devIndex);

//       // if the bandwidth drop too much, clear the slide window
//       unsigned long long sendTime = _log.diff - slideWindow[devIndex].front().diff;
//       unsigned long long sendDataSizes = windowDataSizes[devIndex] + _log.size - slideWindow[devIndex].front().size;
//       // 0.93 = 1e9 / 1024 * 1024 * 1024
//       int curr_bandwidths = sendDataSizes * 0.93 * 8 / sendTime;
//       if (curr_bandwidths < prev_bandwidths / 2) {
//         // clear slide window
//         emptySlideWindow(devIndex);
//         windowDataSizes[devIndex] = 0;
//       }
//       maxNcclFuncTimes = _log.ncclFuncTimes;
//     }
//     if (slideWindow[devIndex].size() >= maxWindowSize) {
//       timer_log frontLog = slideWindow[devIndex].front();
//       slideWindow[devIndex].pop();
//       windowDataSizes[devIndex] -= frontLog.size;
//     }
//     slideWindow[devIndex].push(_log);
//     sendEndTime[devIndex] = _log.diff;
//     windowDataSizes[devIndex] += _log.size;
//     return;
//   }

//   void popSlideWindow(int devIndex) {
//     if (!slideWindow[devIndex].empty()) {
//       slideWindow[devIndex].pop();
//     }
//     return;
//   }

//   void emptySlideWindow(int devIndex) {
//     while (!slideWindow[devIndex].empty()) {
//       slideWindow[devIndex].pop();
//     }
//     return;
//   }

//   int getBandWidths(int devIndex) {
//     if (slideWindow[devIndex].size() < maxWindowSize) {
//       return 0;
//     }

//     // Gbps
//     unsigned long long sendTime = sendEndTime[devIndex] - slideWindow[devIndex].front().diff;
//     unsigned long long sendDataSizes = windowDataSizes[devIndex] - slideWindow[devIndex].front().size;
//     // 0.93 = 1e9 / 1024 * 1024 * 1024
//     return sendDataSizes * 0.93 * 8 / sendTime;
//   }

//   void init(){
//     std::lock_guard<std::mutex> stateLock(telemetryStateLock);
//     if(live == -1){
//       state = 0;
//       stop = 0;
//       windowDataSizes[0] = 0;
//       windowDataSizes[1] = 0;
//       windowDataSizes[2] = 0;
//       windowDataSizes[3] = 0;
//       collect = 0;
//       live = 1;
//       pthread_mutex_init(&lock, NULL);
//       pthread_create(&thread, NULL, timerLogService, NULL);
//     }
//   }

//   bool setState(int to){
//     // return 1;

//     if(state != 0) return 0;
//     if(pthread_mutex_trylock(&lock) == 0){
//       if(state == 0) state = to;
//       pthread_mutex_unlock(&lock);
//       return state == to;
//     }
//     return 0;
//   } 

//   void freeState(){
//     // pthread_mutex_lock(&lock);
//     state = 0;
//     // pthread_mutex_unlock(&lock);
//   }

//   void destroy(){
//     std::lock_guard<std::mutex> stateLock(telemetryStateLock);
//     if(live == 1){
//       stop = 1;
//       pthread_join(thread, nullptr);
//       live = 0;
//     }
//   }
// };

// #ifdef NET_IB_CC
// struct timer_log_queue global_timer_log;
// #else
// extern struct timer_log_queue global_timer_log;
// #endif



// /*diff net_ib.cc:
//  *#define NET_IB_CC
//  *#include "timer_log.h"
//  *proxy.cc:
//  *pthread_create(&global_timer_log.thread, NULL, timerLogService, NULL);
//  */

// #endif
