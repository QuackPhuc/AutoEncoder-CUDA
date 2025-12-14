#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>

// Simple logger for training metrics with optional file output
class Logger {
public:
    // logFile: empty string for console-only logging
    Logger(const std::string& logFile = "");
    ~Logger();
    
    // Log epoch progress: epoch/totalEpochs, loss value, and elapsed time
    void logEpoch(int epoch, int totalEpochs, float loss, double timeSeconds);
    void log(const std::string& message);

private:
    std::ofstream m_logFile;
    bool m_fileLogging;
};

#endif // LOGGER_H
