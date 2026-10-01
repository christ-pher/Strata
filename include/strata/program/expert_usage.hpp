#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::program {
// Main-model decode/verification routing only: no extra device transfers, prompt text, or token IDs.
// The caller owns the counters. One bounded snapshot is handed to the writer at request boundaries.
class ExpertUsage {
public:
    ~ExpertUsage() {
        if (!writer_.joinable()) return;
        publish();
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_one();
        writer_.join();
    }
    bool start(const std::string& directory, int layers, int experts) {
        if (layers <= 0 || experts <= 0) return false;
        try {
            std::filesystem::create_directories(directory);
            const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
            path_ = std::filesystem::path(directory) / ("usage-" + std::to_string(stamp) + ".tsv");
            layers_ = layers; experts_ = experts;
            counts_.assign(size_t(layers) * experts, 0);
            // Detect an unwritable destination before enabling recording.
            if (!write(counts_)) return false;
            writer_ = std::thread([this] { run(); });
            return true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata usage: recording disabled: %s\n", e.what());
            return false;
        }
    }
    void record(int64_t layer, const int32_t* ids, int64_t tokens, int64_t k) {
        if (!writer_.joinable() || layer < 0 || layer >= layers_) return;
        for (int64_t i = 0; i < tokens * k; ++i) {
            const int32_t expert = ids[i];
            if (expert >= 0 && expert < experts_) ++counts_[size_t(layer) * experts_ + expert];
        }
    }
    void publish() {
        if (!writer_.joinable()) return;
        { std::lock_guard lock(mutex_); pending_ = counts_; dirty_ = true; }
        ready_.notify_one();
    }
private:
    bool write(const std::vector<uint64_t>& counts) {
        const auto temporary = path_.string() + ".tmp";
        std::ofstream out(temporary, std::ios::trunc);
        out << "# strata-expert-usage-v1 " << layers_ << ' ' << experts_ << '\n';
        for (int l = 0; l < layers_; ++l) for (int e = 0; e < experts_; ++e) {
            const auto n = counts[size_t(l) * experts_ + e];
            if (n) out << l << '\t' << e << '\t' << n << '\n';
        }
        out.close();
        if (!out) return false;
        std::error_code error;
        std::filesystem::rename(temporary, path_, error);
        // Windows does not replace an existing destination. Recording is intended for the Linux launcher.
        return !error;
    }
    void run() {
        for (;;) {
            std::vector<uint64_t> snapshot;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return dirty_ || stopping_; });
                if (!dirty_) return;
                snapshot.swap(pending_); dirty_ = false;
            }
            if (!write(snapshot)) std::fprintf(stderr, "strata usage: cannot save %s\n", path_.string().c_str());
        }
    }
    int layers_ = 0, experts_ = 0;
    std::filesystem::path path_;
    std::vector<uint64_t> counts_, pending_;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool dirty_ = false, stopping_ = false;
    std::thread writer_;
};
} // namespace strata::program
