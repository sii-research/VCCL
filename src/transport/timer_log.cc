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

struct TelemetryBandwidthInfo {
  unsigned long long startTime;
  int maxBandwidths[ncclNumFuncs][2];
  int meanBandwidths[ncclNumFuncs][2];
  int bandwidthCounts[ncclNumFuncs][2];
};
static TelemetryBandwidthInfo telemetryBandwidthInfo;
static unsigned long long pinpointDuration = 100 * 1000 * 1000; // 100 ms
static int previousMeanBandwidths[ncclNumFuncs][2] = {0};
static bool occurAbnormal[ncclNumFuncs][2] = {false};

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
                                     hostname + "_" + ncName + "_Port" + std::to_string(log.devIndex) +
                                     (i == 0 ? "_A.log" : "_B.log");
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
          // update slide winrow
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
                                     hostname + "_" + ncName + "_Port" + std::to_string(log.devIndex) +
                                     (i == 0 ? "_A.log" : "_B.log");
              portLogs.filenames[i] = filename;
              portLogs.files[i].open(filename, std::ios::trunc);
              portLogs.files[i] << "Time,DevIndex,Func,MaxBandwidth,MeanBandwidth,Status\n";
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
            *pFile << "Time,DevIndex,Func,MaxBandwidth,MeanBandwidth,Status\n";
            portLogs.headerWritten[portLogs.currentFile] = true;
          }
          int bandWidths = global_timer_log.getBandWidths(log.devIndex);
          char dataBuffer[512];
          unsigned long long current_timestamp = log.diff;
          if (telemetryBandwidthInfo.startTime == 0) {
            telemetryBandwidthInfo.startTime = current_timestamp;
            telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex] = bandWidths;
            telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] = bandWidths;
            telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex] = 1;
          }
          else {
            if (bandWidths > telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex]) {
              telemetryBandwidthInfo.maxBandwidths[log.func][log.devIndex] = bandWidths;
            }
            telemetryBandwidthInfo.meanBandwidths[log.func][log.devIndex] += bandWidths;
            telemetryBandwidthInfo.bandwidthCounts[log.func][log.devIndex] += 1;

            // judge if need to output due to duration
            if (current_timestamp - telemetryBandwidthInfo.startTime < pinpointDuration) {
              continue;
            }
            // output previous duration info
            for (int devIndex = 0; devIndex < 2; devIndex++) {
              for (int ncclFuncIndex = 0; ncclFuncIndex < ncclNumFuncs; ncclFuncIndex++) {
                if (telemetryBandwidthInfo.bandwidthCounts[ncclFuncIndex][devIndex] == 0) {
                  continue;
                }
                int meanBandwidth = telemetryBandwidthInfo.meanBandwidths[ncclFuncIndex][devIndex] /
                                    telemetryBandwidthInfo.bandwidthCounts[ncclFuncIndex][devIndex];

                memset(dataBuffer, 0, sizeof(dataBuffer));
                /*sprintf(dataBuffer, "%s,%d,%u,%d,%d,%d",
                        getCurrentTimeString().c_str(), devIndex,
                        ncclFuncIndex,
                        telemetryBandwidthInfo.maxBandwidths[ncclFuncIndex][devIndex],
                        meanBandwidth,
                        0);
                (*pFile) << dataBuffer << std::endl;*/

                // judge if exists exception
                if (!occurAbnormal[ncclFuncIndex][devIndex] && meanBandwidth < previousMeanBandwidths[ncclFuncIndex][devIndex] / 2) {
                  // exception
                  PortLogs &portLogs = logFilesMap[devIndex];
                  std::ofstream *pFile = &portLogs.files[portLogs.currentFile];
                  sprintf(dataBuffer, "%s,%d,%u,%d,%d,%d",
                          getCurrentTimeString().c_str(), devIndex,
                          ncclFuncIndex,
                          telemetryBandwidthInfo.maxBandwidths[ncclFuncIndex][devIndex],
                          meanBandwidth,
                          1);
                  (*pFile) << dataBuffer << std::endl;
                  occurAbnormal[ncclFuncIndex][devIndex] = true;
                }
                else if (meanBandwidth >= previousMeanBandwidths[ncclFuncIndex][devIndex] / 2){
                  occurAbnormal[ncclFuncIndex][devIndex] = false;
                }
                previousMeanBandwidths[ncclFuncIndex][devIndex] = std::max(meanBandwidth, previousMeanBandwidths[ncclFuncIndex][devIndex]);

                // reset for new duration
                telemetryBandwidthInfo.startTime = 0;
                telemetryBandwidthInfo.maxBandwidths[ncclFuncIndex][devIndex] = 0;
                telemetryBandwidthInfo.meanBandwidths[ncclFuncIndex][devIndex] = 0;
                telemetryBandwidthInfo.bandwidthCounts[ncclFuncIndex][devIndex] = 0;
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