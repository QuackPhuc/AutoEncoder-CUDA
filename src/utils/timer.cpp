#include "timer.h"

Timer::Timer() : m_running(false) {}

void Timer::start() {
    m_startTime = std::chrono::steady_clock::now();
    m_running = true;
}

double Timer::stop() {
    m_endTime = std::chrono::steady_clock::now();
    m_running = false;
    return elapsed();
}

double Timer::elapsed() const {
    auto endTime = m_running ? std::chrono::steady_clock::now() : m_endTime;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime);
    return duration.count() / 1000000.0;  // microseconds to seconds
}
