// Logger.h
#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>

namespace forte2 {
class Logger {
  public:
    enum Level { NONE = 0, ERROR = 1, WARNING = 2, INFO = 3, DEBUG = 4 };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(Level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_level_ = level;
    }

    Level getLevel() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_level_;
    }

    void log(Level level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level <= current_level_) { // Changed: now <= instead of >=
            auto& stream = (level == ERROR) ? std::cerr : std::cout;
            stream << "[" << levelToString(level) << "] " << message << std::endl;
        }
    }

  private:
    Logger() = default;
    Level current_level_ = INFO; // Default to INFO level
    mutable std::mutex mutex_;

    std::string levelToString(Level level) const {
        switch (level) {
        case NONE:
            return "NONE";
        case ERROR:
            return "ERROR";
        case WARNING:
            return "WARNING";
        case INFO:
            return "INFO";
        case DEBUG:
            return "DEBUG";
        default:
            return "UNKNOWN";
        }
    }
};

class LogStream {
  private:
    std::ostringstream oss_;
    Logger::Level level_;
    bool should_log_;

  public:
    LogStream(Logger::Level level)
        : level_(level), should_log_(level <= Logger::getInstance().getLevel()) {
    } // Changed: <= instead of >=

    template <typename T> LogStream& operator<<(const T& value) {
        if (should_log_) {
            oss_ << value;
        }
        return *this;
    }

    ~LogStream() {
        if (should_log_) {
            Logger::getInstance().log(level_, oss_.str());
        }
    }

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    LogStream(LogStream&& other) noexcept
        : oss_(std::move(other.oss_)), level_(other.level_), should_log_(other.should_log_) {
        other.should_log_ = false;
    }
};

#define LOG_ERROR LogStream(Logger::ERROR)
#define LOG_WARNING LogStream(Logger::WARNING)
#define LOG_INFO LogStream(Logger::INFO)
#define LOG_DEBUG LogStream(Logger::DEBUG)
} // namespace forte2