#include "logger.h"
#include <iostream>
#include <iomanip>

Logger::Logger(const std::string& logFile) : m_fileLogging(false) {
    if (!logFile.empty()) {
        m_logFile.open(logFile);
        if (m_logFile.is_open()) {
            m_fileLogging = true;
        } else {
            std::cerr << "Warning: Could not open log file '" << logFile 
                      << "'. Logging to console only." << std::endl;
        }
    }
}

Logger::~Logger() {
    if (m_logFile.is_open()) {
        m_logFile.close();
    }
}

void Logger::logEpoch(int epoch, int totalEpochs, float loss, double timeSeconds) {
    std::string message = "Epoch [" + std::to_string(epoch) + "/" + 
                         std::to_string(totalEpochs) + "] | Loss: " + 
                         std::to_string(loss) + " | Time: " + 
                         std::to_string(timeSeconds) + "s";
    
    std::cout << message << std::endl;
    if (m_fileLogging) {
        m_logFile << message << std::endl;
        m_logFile.flush();
    }
}

void Logger::log(const std::string& message) {
    std::cout << message << std::endl;
    if (m_fileLogging) {
        m_logFile << message << std::endl;
        m_logFile.flush();
    }
}
