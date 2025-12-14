#ifndef TIMER_H
#define TIMER_H

#include <chrono>

// Monotonic timer for performance measurement (uses steady_clock)
class Timer {
public:
    Timer();
    
    void start();
    double stop();     // Stop and return elapsed seconds
    double elapsed() const;  // Get elapsed without stopping

private:
    std::chrono::steady_clock::time_point m_startTime;
    std::chrono::steady_clock::time_point m_endTime;
    bool m_running;
};

#endif // TIMER_H
