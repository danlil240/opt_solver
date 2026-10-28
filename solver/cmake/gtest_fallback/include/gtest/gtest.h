#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace testing {

class AssertionResult {
public:
    explicit AssertionResult(bool ok) : ok_(ok) {}
    explicit operator bool() const noexcept { return ok_; }
    template <typename T>
    AssertionResult& operator<<(const T& val) {
        std::ostringstream oss;
        oss << val;
        msg_ += oss.str();
        return *this;
    }
    const std::string& message() const noexcept { return msg_; }
private:
    bool ok_;
    std::string msg_;
};

class Test {
public:
    virtual ~Test() = default;
    virtual void SetUp() {}
    virtual void TearDown() {}
    virtual void TestBody() = 0;
};

struct TestEntry {
    std::string suite;
    std::string name;
    std::function<Test*()> factory;
};

struct TestContext {
    int failures_in_current_test = 0;
    bool current_test_skipped = false;
    bool abort_current_test = false;
    std::string filter = "*";
    bool list_tests = false;
    std::vector<std::string> scoped_traces;
    std::vector<TestEntry> tests;

    static TestContext& instance() {
        static TestContext ctx;
        return ctx;
    }
};

class MessageSink {
public:
    MessageSink(const char* file, int line, const std::string& expr, bool fatal, bool skip = false)
        : file_(file), line_(line), expr_(expr), fatal_(fatal), skip_(skip) {}

    ~MessageSink() noexcept(false) {
        auto& ctx = TestContext::instance();
        if (skip_) {
            ctx.current_test_skipped = true;
            std::printf("[  SKIPPED ] %s:%d: %s\n", file_, line_, oss_.str().c_str());
            return;
        }
        ++ctx.failures_in_current_test;
        std::fprintf(stderr, "%s:%d: Failure\n  %s\n", file_, line_, expr_.c_str());
        if (!oss_.str().empty()) {
            std::fprintf(stderr, "  Message: %s\n", oss_.str().c_str());
        }
        for (const auto& t : ctx.scoped_traces) {
            std::fprintf(stderr, "  Scoped trace: %s\n", t.c_str());
        }
        if (fatal_) {
            ctx.abort_current_test = true;
        }
    }

    template <typename T>
    MessageSink& operator<<(const T& val) {
        oss_ << val;
        return *this;
    }

    MessageSink& operator<<(std::ostream& (*manip)(std::ostream&)) {
        oss_ << manip;
        return *this;
    }

private:
    const char* file_;
    int line_;
    std::string expr_;
    bool fatal_;
    bool skip_;
    std::ostringstream oss_;
};

struct FatalReturnHelper {
    void operator=(const MessageSink&) const {}
};

class ScopedTrace {
public:
    template <typename T>
    ScopedTrace(const char* file, int line, const T& msg) {
        std::ostringstream oss;
        oss << file << ":" << line << ": " << msg;
        TestContext::instance().scoped_traces.push_back(oss.str());
    }
    ~ScopedTrace() {
        auto& st = TestContext::instance().scoped_traces;
        if (!st.empty()) st.pop_back();
    }
};

inline int register_test(const char* suite, const char* name, std::function<Test*()> factory) {
    TestContext::instance().tests.push_back({suite, name, std::move(factory)});
    return 0;
}

inline bool pattern_match(const char* pat, const char* str) {
    while (*pat) {
        if (*pat == '*') {
            ++pat;
            if (!*pat) return true;
            while (*str) {
                if (pattern_match(pat, str)) return true;
                ++str;
            }
            return false;
        } else if (*pat == '?' || *pat == *str) {
            if (!*str) return false;
            ++pat;
            ++str;
        } else {
            return false;
        }
    }
    return *str == '\0';
}

inline bool matches_filter(const std::string& filter, const std::string& full_name) {
    if (filter.empty() || filter == "*") return true;
    std::string pos_patterns = filter;
    std::string neg_patterns;
    auto dash = filter.find('-');
    if (dash != std::string::npos) {
        pos_patterns = filter.substr(0, dash);
        neg_patterns = filter.substr(dash + 1);
    }
    auto match_any = [&](const std::string& list) {
        std::size_t start = 0;
        while (start <= list.size()) {
            auto colon = list.find(':', start);
            std::string sub = (colon == std::string::npos) ? list.substr(start) : list.substr(start, colon - start);
            if (!sub.empty() && pattern_match(sub.c_str(), full_name.c_str())) return true;
            if (colon == std::string::npos) break;
            start = colon + 1;
        }
        return false;
    };
    if (!pos_patterns.empty() && !match_any(pos_patterns)) return false;
    if (!neg_patterns.empty() && match_any(neg_patterns)) return false;
    return true;
}

inline void InitGoogleTest(int* argc, char** argv) {
    if (!argc || !argv) return;
    auto& ctx = TestContext::instance();
    for (int i = 1; i < *argc; ++i) {
        const char* arg = argv[i];
        if (!arg) continue;
        if (std::strncmp(arg, "--gtest_filter=", 15) == 0) {
            ctx.filter = arg + 15;
        } else if (std::strcmp(arg, "--gtest_list_tests") == 0) {
            ctx.list_tests = true;
        }
    }
}

namespace detail {
template <typename A, typename B>
bool cmp_eq(const A& a, const B& b) { return a == b; }
template <typename A, typename B>
bool cmp_ne(const A& a, const B& b) { return !(a == b); }
template <typename A, typename B>
bool cmp_lt(const A& a, const B& b) { return a < b; }
template <typename A, typename B>
bool cmp_le(const A& a, const B& b) { return a <= b; }
template <typename A, typename B>
bool cmp_gt(const A& a, const B& b) { return a > b; }
template <typename A, typename B>
bool cmp_ge(const A& a, const B& b) { return a >= b; }

inline bool cmp_double_eq(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (a == b) return true;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return diff <= 4.0 * 2.2204460492503131e-16 * scale;
}

inline bool cmp_near(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::fabs(a - b) <= tol;
}
} // namespace detail

} // namespace testing

inline int RUN_ALL_TESTS() {
    auto& ctx = ::testing::TestContext::instance();
    if (ctx.list_tests) {
        std::string last_suite;
        for (const auto& t : ctx.tests) {
            if (t.suite != last_suite) {
                std::printf("%s.\n", t.suite.c_str());
                last_suite = t.suite;
            }
            std::printf("  %s\n", t.name.c_str());
        }
        return 0;
    }

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    for (const auto& t : ctx.tests) {
        std::string full = t.suite + "." + t.name;
        if (!::testing::matches_filter(ctx.filter, full)) continue;

        ctx.failures_in_current_test = 0;
        ctx.current_test_skipped = false;
        ctx.abort_current_test = false;
        ctx.scoped_traces.clear();

        std::printf("[ RUN      ] %s\n", full.c_str());
        std::fflush(stdout);
        try {
            std::unique_ptr<::testing::Test> inst(t.factory());
            inst->SetUp();
            if (!ctx.abort_current_test && !ctx.current_test_skipped) {
                inst->TestBody();
            }
            inst->TearDown();
        } catch (const std::exception& e) {
            ++ctx.failures_in_current_test;
            std::fprintf(stderr, "Uncaught std::exception in %s: %s\n", full.c_str(), e.what());
        } catch (...) {
            ++ctx.failures_in_current_test;
            std::fprintf(stderr, "Uncaught unknown exception in %s\n", full.c_str());
        }

        if (ctx.current_test_skipped && ctx.failures_in_current_test == 0) {
            ++skipped;
            std::printf("[  SKIPPED ] %s\n", full.c_str());
        } else if (ctx.failures_in_current_test == 0) {
            ++passed;
            std::printf("[       OK ] %s\n", full.c_str());
        } else {
            ++failed;
            std::printf("[  FAILED  ] %s\n", full.c_str());
        }
        std::fflush(stdout);
    }

    std::printf("[==========] %d passed, %d failed, %d skipped.\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}

#define SMF_GTEST_CLASS_NAME_(suite, name) suite##_##name##_Test

#define TEST(suite, name)                                                                          \
    class SMF_GTEST_CLASS_NAME_(suite, name) : public ::testing::Test {                            \
    public:                                                                                        \
        void TestBody() override;                                                                  \
    };                                                                                             \
    static int SMF_GTEST_CLASS_NAME_(suite, name)##_reg_ = ::testing::register_test(               \
        #suite, #name, []() -> ::testing::Test* { return new SMF_GTEST_CLASS_NAME_(suite, name)(); }); \
    void SMF_GTEST_CLASS_NAME_(suite, name)::TestBody()

#define TEST_F(fixture, name)                                                                      \
    class SMF_GTEST_CLASS_NAME_(fixture, name) : public fixture {                                  \
    public:                                                                                        \
        void TestBody() override;                                                                  \
    };                                                                                             \
    static int SMF_GTEST_CLASS_NAME_(fixture, name)##_reg_ = ::testing::register_test(             \
        #fixture, #name, []() -> ::testing::Test* { return new SMF_GTEST_CLASS_NAME_(fixture, name)(); }); \
    void SMF_GTEST_CLASS_NAME_(fixture, name)::TestBody()

#define EXPECT_TRUE(cond)                                                                          \
    switch (0) case 0: default:                                                                    \
    if (static_cast<bool>(cond)) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #cond " is true", false)

#define EXPECT_FALSE(cond)                                                                         \
    switch (0) case 0: default:                                                                    \
    if (!static_cast<bool>(cond)) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #cond " is false", false)

#define ASSERT_TRUE(cond)                                                                          \
    switch (0) case 0: default:                                                                    \
    if (static_cast<bool>(cond)) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #cond " is true", true)

#define ASSERT_FALSE(cond)                                                                         \
    switch (0) case 0: default:                                                                    \
    if (!static_cast<bool>(cond)) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #cond " is false", true)

#define EXPECT_EQ(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_eq((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " == " #b, false)

#define EXPECT_NE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_ne((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " != " #b, false)

#define EXPECT_LT(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_lt((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " < " #b, false)

#define EXPECT_LE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_le((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " <= " #b, false)

#define EXPECT_GT(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_gt((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " > " #b, false)

#define EXPECT_GE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_ge((a), (b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected: " #a " >= " #b, false)

#define ASSERT_EQ(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_eq((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " == " #b, true)

#define ASSERT_NE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_ne((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " != " #b, true)

#define ASSERT_LT(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_lt((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " < " #b, true)

#define ASSERT_LE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_le((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " <= " #b, true)

#define ASSERT_GT(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_gt((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " > " #b, true)

#define ASSERT_GE(a, b)                                                                            \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_ge((a), (b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion failed: " #a " >= " #b, true)

#define EXPECT_DOUBLE_EQ(a, b)                                                                     \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_double_eq(static_cast<double>(a), static_cast<double>(b))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected double eq: " #a " ~= " #b, false)

#define ASSERT_DOUBLE_EQ(a, b)                                                                     \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_double_eq(static_cast<double>(a), static_cast<double>(b))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion double eq failed: " #a " ~= " #b, true)

#define EXPECT_NEAR(a, b, tol)                                                                     \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_near(static_cast<double>(a), static_cast<double>(b), static_cast<double>(tol))) {} else ::testing::MessageSink(__FILE__, __LINE__, "Expected near: |" #a " - " #b "| <= " #tol, false)

#define ASSERT_NEAR(a, b, tol)                                                                     \
    switch (0) case 0: default:                                                                    \
    if (::testing::detail::cmp_near(static_cast<double>(a), static_cast<double>(b), static_cast<double>(tol))) {} else return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion near failed: |" #a " - " #b "| <= " #tol, true)

#define EXPECT_NO_THROW(stmt)                                                                      \
    do { try { stmt; } catch (...) { ::testing::MessageSink(__FILE__, __LINE__, "Expected no throw: " #stmt, false); } } while (0)

#define ASSERT_NO_THROW(stmt)                                                                      \
    do { try { stmt; } catch (...) { return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "Assertion no throw failed: " #stmt, true); } } while (0)

#define EXPECT_THROW(stmt, exc_type)                                                               \
    do { bool caught_ = false; try { stmt; } catch (const exc_type&) { caught_ = true; } catch (...) {} if (!caught_) ::testing::MessageSink(__FILE__, __LINE__, "Expected throw: " #stmt, false); } while (0)

#define EXPECT_NO_FATAL_FAILURE(stmt)                                                              \
    do { stmt; } while (0)

#define ASSERT_NO_FATAL_FAILURE(stmt)                                                              \
    do { stmt; if (::testing::TestContext::instance().abort_current_test) return; } while (0)

#define SUCCEED() do {} while (0)
#define FAIL() return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "FAIL()", true)
#define ADD_FAILURE() ::testing::MessageSink(__FILE__, __LINE__, "ADD_FAILURE()", false)
#define GTEST_SKIP() return ::testing::FatalReturnHelper() = ::testing::MessageSink(__FILE__, __LINE__, "SKIPPED", false, true)

#define SMF_GTEST_CONCAT_INNER_(a, b) a##b
#define SMF_GTEST_CONCAT_(a, b) SMF_GTEST_CONCAT_INNER_(a, b)
#define SCOPED_TRACE(msg) ::testing::ScopedTrace SMF_GTEST_CONCAT_(smf_scoped_trace_, __LINE__)(__FILE__, __LINE__, (msg))
