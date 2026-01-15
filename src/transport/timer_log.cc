#include "timer_log.h"
#include "nccl.h"
#include "core.h"
#include <sys/un.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <queue>

const int nccl_telemetry_enable = ncclGetEnv("NCCL_TELEMETRY_ENABLE") ? atoi(ncclGetEnv("NCCL_TELEMETRY_ENABLE")) : 0;
const char* nccl_telemetry_log_path = ncclGetEnv("NCCL_TELEMETRY_LOG_PATH");
const int nccl_telemetry_observe = ncclGetEnv("NCCL_TELEMETRY_OBSERVE") ? atoi(ncclGetEnv("NCCL_TELEMETRY_OBSERVE")) : 0;

// void sigpipe_handler(int signum) {
//   ;
// }

static int getPortCount() {
  const char* env = ncclGetEnv("NCCL_NUM_PORTS");
  if (env) {
    int n = atoi(env);
    if (n == 1 || n == 2) return n;
  }
  return 2; // default
}

std::string getCurrentTimeString() {
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y%m%d%H%M%S");
    return oss.str();
}

void printLogInfo(struct timer_log log){
  INFO(NCCL_NET, "%d.%d.%d.%d->%d.%d.%d.%d send %d Bits used %lld nsec", 
             log.srcIp[0],log.srcIp[1],log.srcIp[2],log.srcIp[3],
             log.dscIp[0],log.dscIp[1],log.dscIp[2],log.dscIp[3],
             log.size,log.diff
  );
}

struct PortLogs {
  std::ofstream files[2];                 // A-file[0], B-file[1]
  std::string filenames[2];
  int currentFile = 0;                    // 0->A, 1->B
  bool headerWritten[2] = {false, false};
  bool initialized = false;
};
static std::vector<PortLogs> logFilesMap;                            // two local ports
static constexpr size_t MAX_LOG_SIZE = 10 * 1024 * 1024;            // 10MB
static unsigned long long logFilesStartTime = 0;
static unsigned long long logFileDuration = 5 * 60 * (unsigned long long)1000000000; // 5min in nanoseconds

struct TelemetryBandwidthInfo {
  unsigned long long startTime;
  int maxBandwidths[ncclNumFuncs][2];
  int meanBandwidths[ncclNumFuncs][2];
  int bandwidthCounts[ncclNumFuncs][2];
  // Used for trace previous pinpointDuration logs
  std::queue<timer_log> previousLogs[2];
};

// Used for pinpoint network abnormalities
static TelemetryBandwidthInfo telemetryBandwidthInfo;
static int pinpointDuration = 100 * 1000 * 1000; // 100 ms
// static int previousMeanBandwidths[ncclNumFuncs][2] = {0};
static bool occurAbnormal[2] = {false};
static unsigned long long lastAbnormalTime[2] = {0};
static int normalPreviousBandwidths[ncclNumFuncs][2] = {0};

// Used for trace average network bandwidths every traceDuration
static TelemetryBandwidthInfo telemetryAverageBandwidthInfo;
static int traceDuration = 1 * 1000 * 1000 * 1000; // 1 second

// Used for compress data for bandwidth/status/timestamp for logging
static const uint64_t TIME_BITS = 54;
static const uint64_t BANDWIDTH_BITS = 9;
static const uint64_t STATUS_BITS = 1;
static const uint64_t TIME_MASK = (1ULL << TIME_BITS) - 1;
static const uint64_t BANDWIDTH_MASK = (1ULL << BANDWIDTH_BITS) - 1;
// Basetime for compressing timestamp
// We set basetime for the first log
static uint64_t BASETIME = 0; // in nanoseconds
// Compress function
uint64_t compress_func(uint64_t timestamp, int bandwidth, int status) {
  // 1. Calculate delta time
  uint64_t delta_time = timestamp - BASETIME;
  if (delta_time > TIME_MASK) {
    std::cerr << "Timestamp exceeds compressible range!" << std::endl;
    return 0;
  }

  uint64_t res = 0;

  // 2. Set time bits
  res |= (delta_time & TIME_MASK);

  // 3. Set bandwidth bits
  res |= ((static_cast<uint64_t>(bandwidth) & BANDWIDTH_MASK) << TIME_BITS);

  // 4. Set status bit
  res |= ((static_cast<uint64_t>(status) & 0x1) << (TIME_BITS + BANDWIDTH_BITS));
  return res;
}

void* timerLogService(void *args){
  // signal(SIGPIPE, sigpipe_handler);
  //setupTelemetry();//set up environment variables
  struct sockaddr_un server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sun_family = AF_UNIX;
  strncpy(server_addr.sun_path, SOCK_PATH, sizeof(server_addr.sun_path) - 1);
  //WARN("------------NCCL_TELEMETRY_ENABLE = %d-------------", nccl_telemetry_enable);

  if(TIMER_LOG_NCCL_TELEMETRY){
    // char buffer[256];
    const int TOTAL_PORTS = getPortCount();
    logFilesMap.resize(TOTAL_PORTS);
    //std::string baseName = global_timer_log.log[i].NetworkCardName;
    std::string timestamp = getCurrentTimeString();

    while(!global_timer_log.stop){
      global_timer_log.collect = 1;

      __sync_synchronize();
      if(!global_timer_log.log.empty()){

        __sync_synchronize();
        if(global_timer_log.log.empty()){
          continue;
        }

        if (nccl_telemetry_observe) {
          timer_log log = global_timer_log.pop();

          if (log.diff == 0) {
            continue;
          }

          // update slide window
          global_timer_log.pushSlideWindow(log, log.devIndex);
          if (global_timer_log.slideWindow[log.devIndex].size() < maxWindowSize) {
            continue;
          }

          /* save log */
          if (log.devIndex >= logFilesMap.size())
            logFilesMap.resize(log.devIndex + 1);

          PortLogs &portLogs = logFilesMap[log.devIndex];
          if (!portLogs.initialized) {
            std::string ncName = log.NetworkCardName;
            char hostname[1024];
            getHostName(hostname, 1024, '.');

            for (int i = 0; i < 2; i++) {
              std::string filename = std::string(nccl_telemetry_log_path) + "/" +
                                     hostname + "-" + ncName + "-Port" + std::to_string(log.devIndex) +
                                     (i == 0 ? "-A.log" : "-B.log");
              portLogs.filenames[i] = filename;
              portLogs.files[i].open(filename, std::ios::trunc);
              portLogs.files[i] << "Time,Group,FromRank,ToRank,DevIndex,Func,FuncTimes,SrcIP,DstIP,Bandwidth,SendWrCounter,RemainWrDataSize,Timestamp\n";
              portLogs.headerWritten[i] = true;
            }
            portLogs.currentFile = 0; // init current file index
            portLogs.initialized = true;
          }

          std::ofstream *pFile = &portLogs.files[portLogs.currentFile];

          if (static_cast<size_t>(pFile->tellp()) >= 10 * 1024 * 1024) {
            pFile->close();
            portLogs.currentFile ^= 1;
            pFile = &portLogs.files[portLogs.currentFile];

            pFile->open(portLogs.filenames[portLogs.currentFile], std::ios::trunc);
            *pFile << "Time,Group,FromRank,ToRank,DevIndex,Func,FuncTimes,SrcIP,DstIP,Bandwidth,SendWrCounter,RemainWrDataSize,Timestamp\n";
            portLogs.headerWritten[portLogs.currentFile] = true;
          }
          int bandWidths = global_timer_log.getBandWidths(log.devIndex);
          char dataBuffer[512];
          sprintf(dataBuffer, "%s,%lu,%d,%d,%d,%u,%lld,%d.%d.%d.%d,%d.%d.%d.%d,%d,%d,%d,%lld",
                  getCurrentTimeString().c_str(), log.groupHash, log.rank, log.peerRank, log.devIndex,
                  log.func, log.ncclFuncTimes,
                  log.srcIp[0], log.srcIp[1], log.srcIp[2], log.srcIp[3],
                  log.dscIp[0], log.dscIp[1], log.dscIp[2], log.dscIp[3],
                  bandWidths, log.sendWrCounter, log.remainWrDataSize, log.diff);
          (*pFile) << dataBuffer << std::endl;
        }
        else {
          timer_log log = global_timer_log.pop();
          if (log.diff == 0) {
            continue;
          }
          // update slide window
          global_timer_log.pushSlideWindow(log, log.devIndex);
          if (global_timer_log.slideWindow[log.devIndex].size() < maxWindowSize) {
            continue;
          }

          /* save log */
          if (log.devIndex >= logFilesMap.size())
            logFilesMap.resize(log.devIndex + 1);

          // Set startTime
          if (BASETIME == 0) BASETIME = log.diff; 

          PortLogs &portLogs = logFilesMap[log.devIndex];
          if (!portLogs.initialized) {
            std::string ncName = log.NetworkCardName;
            char hostname[1024];
            getHostName(hostname, 1024, '.');

            for (int i = 0; i < 2; i++) {
              std::string filename = std::string(nccl_telemetry_log_path) + "/" +
                                     hostname + "-" + ncName + "-Port" + std::to_string(log.devIndex) +
                                     (i == 0 ? "-A.log" : "-B.log");
              portLogs.filenames[i] = filename;
              portLogs.files[i].open(filename, std::ios::trunc);
              portLogs.files[i] << "64 bits data: timestamp(63-10) | bandwidth(9-1) | status(0) \n";
              portLogs.headerWritten[i] = true;
            }
            portLogs.currentFile = 0; // init current file index
            portLogs.initialized = true;
          }

          std::ofstream *pFile = &portLogs.files[portLogs.currentFile];

          if (log.diff - logFilesStartTime >= logFileDuration && static_cast<size_t>(pFile->tellp()) >= 10 * 1024 * 1024) {
            pFile->close();
            portLogs.currentFile ^= 1;
            pFile = &portLogs.files[portLogs.currentFile];

            pFile->open(portLogs.filenames[portLogs.currentFile], std::ios::trunc);
            *pFile << "64 bits data: timestamp(63-10) | bandwidth(9-1) | status(0) \n";
            portLogs.headerWritten[portLogs.currentFile] = true;
            logFilesStartTime = log.diff;
          }
          int bandWidths = global_timer_log.getBandWidths(log.devIndex);
          log.bandwidth = bandWidths;
          unsigned long long current_timestamp = log.diff;

          bool iflog = false;
          // First, pinpoint network abnormalities, if find bandwidth drop too much, output immediately
          // Additionally, if we find bandwidth up again, we also output the logs
          if (telemetryBandwidthInfo.startTime == 0) {
            telemetryBandwidthInfo.startTime = current_timestamp;
            telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex] = bandWidths;
            telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] = bandWidths;
            telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex] = 1;
            telemetryBandwidthInfo.previousLogs[log.devIndex] = std::queue<timer_log>();
            telemetryBandwidthInfo.previousLogs[log.devIndex].push(log);
          }
          else {
            if (bandWidths > telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex]) {
              telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex] = bandWidths;
            }
            int meanBandwidth = telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] /
                                telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex];
            telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] += bandWidths;
            telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex] += 1;
            telemetryBandwidthInfo.previousLogs[log.devIndex].push(log);

            // 1. judge if maintain logs exceeds pinpointDuration
            while (!telemetryBandwidthInfo.previousLogs[log.devIndex].empty()) {
              timer_log &frontLog = telemetryBandwidthInfo.previousLogs[log.devIndex].front();
              if (current_timestamp - frontLog.diff > pinpointDuration) {
                telemetryBandwidthInfo.previousLogs[log.devIndex].pop();
                telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] -= frontLog.bandwidth;
                telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex] -= 1;
              } else {
                break;
              }
            }
            telemetryBandwidthInfo.startTime = telemetryBandwidthInfo.previousLogs[log.devIndex].empty() ?
                                               0 : telemetryBandwidthInfo.previousLogs[log.devIndex].front().diff;
            
            
            // 2. Output after 100ms log after abnormality, if new shift happens, update the last abnormal time
            if (current_timestamp - lastAbnormalTime[log.devIndex] <= pinpointDuration) {
              // output current log
              PortLogs &portLogs = logFilesMap[log.devIndex];
              pFile = &portLogs.files[portLogs.currentFile];

              if (bandWidths < normalPreviousBandwidths[log.func][log.devIndex] / 2) {
                uint64_t compress_log = compress_func(current_timestamp, bandWidths, 1);
                (*pFile) << compress_log << std::endl;
                if (!occurAbnormal[log.devIndex]) {
                  // Detect abnormality
                  occurAbnormal[log.devIndex] = true;
                  lastAbnormalTime[log.devIndex] = log.diff;
                }
              } else {
                uint64_t compress_log = compress_func(current_timestamp, bandWidths, 0);
                (*pFile) << compress_log << std::endl;
                if (occurAbnormal[log.devIndex]) {
                  // Detect recovery
                  occurAbnormal[log.devIndex] = false;
                  lastAbnormalTime[log.devIndex] = log.diff;
                }
              }
              iflog = true;
            }
            // 3. judge if need to output logs
            // output condition: bandwidth drop to half of mean bandwidth
            else if (!occurAbnormal[log.devIndex] && bandWidths < meanBandwidth / 2) {
              // output all maintained logs
              // exception
              PortLogs &portLogs = logFilesMap[log.devIndex];
              pFile = &portLogs.files[portLogs.currentFile];

              unsigned long long lastLogTimestamp = lastAbnormalTime[log.devIndex] + pinpointDuration;
              std::queue<timer_log> tempQueue(telemetryBandwidthInfo.previousLogs[log.devIndex]);
              while (!tempQueue.empty()) {
                timer_log frontLog = tempQueue.front();
                tempQueue.pop();
                if (frontLog.diff > lastLogTimestamp) {
                  uint64_t compressLogs = compress_func(frontLog.diff, frontLog.bandwidth, 0);
                  (*pFile) << compressLogs << std::endl;
                }
              }
              
              uint64_t compress_log = compress_func(current_timestamp, bandWidths, 1);
              (*pFile) << compress_log << std::endl;
              occurAbnormal[log.devIndex] = true;
              lastAbnormalTime[log.devIndex] = current_timestamp;

              // Set normal previous bandwidths for recovering detection
              for (int f = 0; f < ncclNumFuncs; f++) {
                if (telemetryBandwidthInfo.bandwidthCounts[f][log.devIndex] > 0) {
                  normalPreviousBandwidths[f][log.devIndex] = telemetryBandwidthInfo.meanBandwidths[f][log.devIndex] /
                                                            telemetryBandwidthInfo.bandwidthCounts[f][log.devIndex];
                }
              }
              iflog = true;
            }
            // 4. judge if need to output logs
            // output condition: bandwidth recovers to normalPreviousBandwidths
            else if (occurAbnormal[log.devIndex] && bandWidths >= normalPreviousBandwidths[log.func][log.devIndex] / 2) {
              // output current log
              PortLogs &portLogs = logFilesMap[log.devIndex];
              pFile = &portLogs.files[portLogs.currentFile];
              
              unsigned long long lastLogTimestamp = lastAbnormalTime[log.devIndex] + pinpointDuration;
              std::queue<timer_log> tempQueue(telemetryBandwidthInfo.previousLogs[log.devIndex]);
              while (!tempQueue.empty()) {
                timer_log frontLog = tempQueue.front();
                tempQueue.pop();
                if (frontLog.diff > lastLogTimestamp) {
                  uint64_t compressLogs = compress_func(frontLog.diff, frontLog.bandwidth, 0);
                  (*pFile) << compressLogs << std::endl;
                }
              }

              uint64_t compress_log = compress_func(current_timestamp, bandWidths, 0);
              (*pFile) << compress_log << std::endl;
              occurAbnormal[log.devIndex] = false;
              lastAbnormalTime[log.devIndex] = current_timestamp;
              iflog = true;
            }
          }

          // Second, trace average network bandwidths every traceDuration
          if (telemetryAverageBandwidthInfo.startTime == 0) {
            telemetryAverageBandwidthInfo.startTime = current_timestamp;
            telemetryAverageBandwidthInfo.meanBandwidths[log.func][log.devIndex] = bandWidths;
            telemetryAverageBandwidthInfo.bandwidthCounts[log.func][log.devIndex] = 1;
          }
          else if (!iflog) {
            telemetryAverageBandwidthInfo.meanBandwidths[log.func][log.devIndex] += bandWidths;
            telemetryAverageBandwidthInfo.bandwidthCounts[log.func][log.devIndex] += 1;

            if (current_timestamp - telemetryAverageBandwidthInfo.startTime >= traceDuration) {
              for (int devIndex = 0; devIndex < 2; devIndex++) {
                for (int f = 0; f < ncclNumFuncs; f++) {
                  if (telemetryAverageBandwidthInfo.bandwidthCounts[f][devIndex] == 0) {
                    continue;
                  }

                  int avgBandwidth = telemetryAverageBandwidthInfo.meanBandwidths[f][devIndex] /
                                    telemetryAverageBandwidthInfo.bandwidthCounts[f][devIndex];
                  uint64_t compress_log = compress_func(current_timestamp, avgBandwidth, occurAbnormal[devIndex] ? 1 : 0);
                  PortLogs &portLogs = logFilesMap[devIndex];
                  pFile = &portLogs.files[portLogs.currentFile];
                  (*pFile) << compress_log << std::endl;
                  // Ensure each devIndex only output once
                  break;
                }
              }
            }
          }
        }
      }
    }
    for (auto& nic : logFilesMap) {
      nic.files[0].close();
      nic.files[1].close();
    }
  }
  return 0;
}