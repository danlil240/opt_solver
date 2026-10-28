#pragma once
// Minimal in-process sampling profiler (Windows only) for bench_profile.
//
// A background thread repeatedly suspends the profiled thread, records its
// instruction pointer and resumes it. report() symbolizes the samples with
// DbgHelp (needs PDBs: build with /Zi and link with /DEBUG) and prints the
// hottest functions, or source lines with by_line = true.

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

namespace smf::bench {

class Sampler {
public:
    void start() {
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                        &target_, THREAD_ALL_ACCESS, FALSE, 0);
        stop_ = false;
        // Fixed storage: the sampler must not touch the heap while the
        // target is suspended (it may hold the heap lock).
        samples_.assign(kCapacity, 0);
        count_ = 0;
        thread_ = std::thread([this] {
            while (!stop_.load(std::memory_order_relaxed) && count_ < kCapacity) {
                if (SuspendThread(target_) != static_cast<DWORD>(-1)) {
                    CONTEXT ctx{};
                    ctx.ContextFlags = CONTEXT_CONTROL;
                    const bool ok = GetThreadContext(target_, &ctx) != 0;
                    ResumeThread(target_);
                    if (ok) samples_[count_++] = ctx.Rip;
                }
                const auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(50)) {
                }
            }
        });
    }

    void stop() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
        CloseHandle(target_);
    }

    void report(bool by_line, int top = 40) const {
        HANDLE proc = GetCurrentProcess();
        SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
        SymInitialize(proc, nullptr, TRUE);
        std::unordered_map<DWORD64, std::size_t> by_addr;
        for (std::size_t i = 0; i < count_; ++i) ++by_addr[samples_[i]];
        std::map<std::string, std::size_t> agg;
        alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 512];
        for (const auto &[addr, cnt] : by_addr) {
            auto *sym = reinterpret_cast<SYMBOL_INFO *>(buf);
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen = 511;
            DWORD64 disp = 0;
            std::string key = "?";
            if (SymFromAddr(proc, addr, &disp, sym)) key = sym->Name;
            if (by_line) {
                IMAGEHLP_LINE64 line{};
                line.SizeOfStruct = sizeof(line);
                DWORD ldisp = 0;
                if (SymGetLineFromAddr64(proc, addr, &ldisp, &line)) {
                    std::string f = line.FileName;
                    const auto slash = f.find_last_of("\\/");
                    if (slash != std::string::npos) f = f.substr(slash + 1);
                    key = f + ":" + std::to_string(line.LineNumber) + " (" + key + ")";
                }
            }
            agg[key] += cnt;
        }
        std::vector<std::pair<std::size_t, std::string>> rows;
        for (const auto &[k, c] : agg) rows.emplace_back(c, k);
        std::sort(rows.rbegin(), rows.rend());
        const double total = static_cast<double>(count_);
        std::printf("---- %zu samples ----\n", count_);
        for (int i = 0; i < top && i < static_cast<int>(rows.size()); ++i) {
            std::printf("%6.2f%%  %s\n", 100.0 * static_cast<double>(rows[i].first) / total,
                        rows[i].second.c_str());
        }
        SymCleanup(proc);
    }

private:
    HANDLE target_ = nullptr;
    std::atomic<bool> stop_{false};
    std::thread thread_;
    static constexpr std::size_t kCapacity = std::size_t{1} << 22;
    std::vector<DWORD64> samples_;
    std::size_t count_ = 0;
};

} // namespace smf::bench
#endif
