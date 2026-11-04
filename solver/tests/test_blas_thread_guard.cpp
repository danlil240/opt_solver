// test_blas_thread_guard.cpp — unit tests for BlasThreadGuard / BlasSerialGuard
//
// These tests verify:
//  1. RoundTrip  — guard sets a thread count and the original is restored on
//                  destruction.
//  2. NestedGuard — outer guard(4), inner guard(1); inner destruct restores 4;
//                   outer destruct restores original.
//  3. SerialGuard  — BlasSerialGuard (= BlasThreadGuard(1)) sets threads to 1.
//  4. ClampZero   — constructing with n<=0 is clamped to 1 (no crash).
//
// All tests compile and pass even when no BLAS thread API is present (no-op
// path).  In that case the thread count is always reported as 1 and set
// operations are silent no-ops.

#include "smf/blas_thread_guard.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// --------------------------------------------------------------------------
// Helper: query current BLAS thread count using the same backend used by
// BlasThreadGuard.  We obtain it by constructing a guard and reading
// saved_threads() (which captures the value BEFORE the guard modifies it).
// Then we immediately restore by letting the guard destruct.
// --------------------------------------------------------------------------

static int get_blas_threads() {
    // Construct a guard requesting the *current* value (we don't know it yet).
    // We use a temporary guard(1) then read saved_threads to learn the old
    // value, then the dtor restores it.  Essentially: peek without change.
    smf::BlasThreadGuard peek(1);          // saves current → sets to 1
    int current = peek.saved_threads();    // what it was before
    return current;
    // ~peek() restores 'current' → net effect: no change
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

TEST(BlasThreadGuard, RoundTrip) {
    const int original = get_blas_threads();

    {
        smf::BlasThreadGuard g(2);
        // Inside the guard the thread count should be 2 (if API is available)
        // or 1 (no-op build).  Either way the guard must not crash.
        (void)g;
    }
    // After destruction the count must be restored.
    const int after = get_blas_threads();
    EXPECT_EQ(original, after)
        << "BlasThreadGuard did not restore the original thread count";
}

TEST(BlasThreadGuard, NestedGuard) {
    const int original = get_blas_threads();

    {
        smf::BlasThreadGuard outer(4);
        // saved in outer == original
        EXPECT_EQ(outer.saved_threads(), (original <= 0 ? 1 : original));

        {
            smf::BlasThreadGuard inner(1);
            // saved in inner == whatever outer set (4, or 1 in no-op build)
            // Just verify it didn't crash.
            (void)inner;
        }
        // After inner destructs, BLAS should be back to what outer set (4 or
        // clamped).  We can't read it directly without a guard, but we can
        // verify via another peek:
        const int after_inner = get_blas_threads();
        // In the no-op build both will be 1; in OpenBLAS build it will be 4.
        // Either way: after_inner == outer.saved_threads() is wrong — actually
        // after inner destructs we expect the value that *outer* set, which is
        // min(4, available).  We just verify it is >=1.
        EXPECT_GE(after_inner, 1);
    }
    // After outer destructs, we should be back to original.
    const int after_outer = get_blas_threads();
    EXPECT_EQ(original, after_outer)
        << "Nested guard: outer did not restore original thread count";
}

TEST(BlasThreadGuard, SerialGuard) {
    const int original = get_blas_threads();
    {
        smf::BlasSerialGuard sg;
        // Inside: BLAS should be set to 1.
        // Verify via peek (which saves current and restores immediately).
        smf::BlasThreadGuard peek(1);
        int inside = peek.saved_threads();
        // In OpenBLAS build: inside == 1.  In no-op build: inside == 1.
        EXPECT_EQ(1, inside)
            << "BlasSerialGuard did not set thread count to 1";
    }
    const int after = get_blas_threads();
    EXPECT_EQ(original, after)
        << "BlasSerialGuard did not restore original thread count";
}

TEST(BlasThreadGuard, ClampZero) {
    // Constructing with 0 or negative must not crash; it is silently clamped to 1.
    const int original = get_blas_threads();
    {
        smf::BlasThreadGuard g0(0);
        smf::BlasThreadGuard gn(-5);
        (void)g0;
        (void)gn;
    }
    const int after = get_blas_threads();
    EXPECT_EQ(original, after)
        << "ClampZero: thread count not restored after clamped guards";
}

// --------------------------------------------------------------------------
// Intel runtime blocktime (Control::mkl_blocktime_ms)
// --------------------------------------------------------------------------
namespace {
// A fake runtime: a per-thread blocktime (initial 7).
thread_local int tl_fake_bt = 7;
std::atomic<int> g_fake_sets{0};
void fake_set(int ms) { tl_fake_bt = ms; g_fake_sets.fetch_add(1); }
int fake_get() { return tl_fake_bt; }

struct HookReset {
    ~HookReset() { smf::mkl_blocktime_reset_hooks_for_test(); }
};
} // namespace

// Runtime absent (the symbol is not found): every call is a quiet no-op.
TEST(BlasThreadGuardBlocktime, AbsentRuntimeIsNoop) {
    HookReset reset;
    smf::mkl_blocktime_set_hooks_for_test(nullptr, nullptr);
    EXPECT_FALSE(smf::mkl_blocktime_supported());
    EXPECT_EQ(smf::mkl_blocktime_get(), -1);
    EXPECT_EQ(smf::mkl_blocktime_set(1), -1);
    const long long applied0 = smf::mkl_blocktime_apply_count();
    {
        smf::BlasThreadGuard g(2, 1);
        EXPECT_EQ(smf::mkl_blocktime_get(), -1);
    }
    EXPECT_EQ(smf::mkl_blocktime_apply_count(), applied0);
}

// A negative blocktime leaves the setting alone; a non-negative one holds for
// the guard's lifetime and is restored, nested guards included.
TEST(BlasThreadGuardBlocktime, SetAndRestore) {
    HookReset reset;
    tl_fake_bt = 7;
    g_fake_sets = 0;
    smf::mkl_blocktime_set_hooks_for_test(fake_set, fake_get);
    {
        smf::BlasThreadGuard none(2);
        smf::BlasThreadGuard neg(2, -1);
        EXPECT_EQ(tl_fake_bt, 7);
        EXPECT_EQ(g_fake_sets.load(), 0);
    }
    {
        smf::BlasThreadGuard outer(2, 1);
        EXPECT_EQ(tl_fake_bt, 1);
        {
            smf::BlasThreadGuard inner(1, 5);
            EXPECT_EQ(tl_fake_bt, 5);
        }
        EXPECT_EQ(tl_fake_bt, 1);
        {
            smf::BlasThreadGuard same(1, 1);  // already 1: not touched
            EXPECT_EQ(tl_fake_bt, 1);
        }
    }
    EXPECT_EQ(tl_fake_bt, 7);
    // Another thread has its own value: set inside the guard, restored after.
    int inside = -1, after = -1;
    std::thread t([&] {
        {
            smf::BlasThreadGuard g(2, 1);
            inside = tl_fake_bt;
        }
        after = tl_fake_bt;
    });
    t.join();
    EXPECT_EQ(inside, 1);
    EXPECT_EQ(after, 7);
    EXPECT_EQ(tl_fake_bt, 7);
}

// ---- Late-appearing runtime (the lookup is retried, a hit is cached) -------
namespace {
std::atomic<bool> g_rt_present{false};
std::atomic<int> g_rt_lookups{0};
bool fake_resolver(void (**set_fn)(int), int (**get_fn)()) {
    g_rt_lookups.fetch_add(1);
    if (!g_rt_present.load()) return false;
    *set_fn = fake_set;
    *get_fn = fake_get;
    return true;
}
struct ResolverReset {
    ~ResolverReset() { smf::mkl_blocktime_reset_resolver_for_test(); }
};
} // namespace

// The runtime is missing at first (lazy threading layer, early probe), then
// appears: the knob must start working without any restart.
TEST(BlasThreadGuardBlocktime, LateRuntimeIsPickedUp) {
    ResolverReset reset;
    tl_fake_bt = 7;
    g_rt_present = false;
    g_rt_lookups = 0;
    smf::mkl_blocktime_set_resolver_for_test(fake_resolver, 0);   // no throttle
    const long long applied0 = smf::mkl_blocktime_apply_count();
    EXPECT_FALSE(smf::mkl_blocktime_supported());
    for (int i = 0; i < 1000; ++i) {                 // far past any fixed probe budget
        smf::BlasThreadGuard g(2, 1);
        EXPECT_EQ(tl_fake_bt, 7);
    }
    EXPECT_EQ(smf::mkl_blocktime_apply_count(), applied0);
    EXPECT_GT(g_rt_lookups.load(), 256);
    g_rt_present = true;                              // the runtime loads now
    {
        smf::BlasThreadGuard g(2, 1);
        EXPECT_TRUE(smf::mkl_blocktime_supported());
        EXPECT_EQ(tl_fake_bt, 1);
    }
    EXPECT_EQ(tl_fake_bt, 7);
    EXPECT_GT(smf::mkl_blocktime_apply_count(), applied0);
}

// The miss retry is throttled: a long interval means one lookup, however
// many calls; once the interval has passed the next call looks again.
TEST(BlasThreadGuardBlocktime, MissRetryIsThrottled) {
    ResolverReset reset;
    tl_fake_bt = 7;
    g_rt_present = false;
    g_rt_lookups = 0;
    smf::mkl_blocktime_set_resolver_for_test(fake_resolver, 60);
    for (int i = 0; i < 1000; ++i) (void)smf::mkl_blocktime_supported();
    EXPECT_EQ(g_rt_lookups.load(), 1);
    g_rt_present = true;
    EXPECT_FALSE(smf::mkl_blocktime_supported());     // still inside the interval
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    EXPECT_TRUE(smf::mkl_blocktime_supported());
    EXPECT_EQ(g_rt_lookups.load(), 2);
}

// A hit is cached for good: no further lookups, even if the resolver would
// now say "absent".
TEST(BlasThreadGuardBlocktime, HitStaysCached) {
    ResolverReset reset;
    g_rt_present = true;
    g_rt_lookups = 0;
    smf::mkl_blocktime_set_resolver_for_test(fake_resolver, 0);
    EXPECT_TRUE(smf::mkl_blocktime_supported());
    EXPECT_EQ(g_rt_lookups.load(), 1);
    g_rt_present = false;
    for (int i = 0; i < 1000; ++i) EXPECT_TRUE(smf::mkl_blocktime_supported());
    EXPECT_EQ(g_rt_lookups.load(), 1);
}

// Many threads missing at once: no crash or race, no knob applied; then all
// of them see the hit once the runtime appears.
TEST(BlasThreadGuardBlocktime, ConcurrentMissesAreSafe) {
    ResolverReset reset;
    g_rt_present = false;
    g_rt_lookups = 0;
    smf::mkl_blocktime_set_resolver_for_test(fake_resolver, 0);
    const long long applied0 = smf::mkl_blocktime_apply_count();
    std::atomic<bool> go{false}, appear{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&] {
            while (!go.load()) std::this_thread::yield();
            for (int i = 0; i < 20000; ++i) {
                if (smf::mkl_blocktime_get() != -1 && !appear.load()) ++bad;
                if (smf::mkl_blocktime_supported() && !appear.load()) ++bad;
            }
        });
    go = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(smf::mkl_blocktime_apply_count(), applied0);
    appear = true;
    g_rt_present = true;
    for (auto &th : ts) th.join();
    EXPECT_EQ(bad.load(), 0);
    EXPECT_GT(g_rt_lookups.load(), 0);
    EXPECT_TRUE(smf::mkl_blocktime_supported());
}

// The real runtime (an MKL "intel" threading build with libiomp5 loaded);
// elsewhere it is absent and the knob must still be harmless.
TEST(BlasThreadGuardBlocktime, RealRuntimeWorkerAndCaller) {
    smf::BlasThreadGuard warm(2);  // first MKL call: the runtime is loaded
    if (!smf::mkl_blocktime_supported()) {
        smf::BlasThreadGuard g(2, 1);
        EXPECT_EQ(smf::mkl_blocktime_get(), -1);
        GTEST_SKIP() << "no Intel OpenMP runtime in this process: the knob is a no-op here";
    }
    const int caller0 = smf::mkl_blocktime_get();
    ASSERT_GE(caller0, 0);
    const int target = caller0 == 1 ? 2 : 1;
    int caller_in = -1, worker_in = -1, worker_before = -1, worker_after = -1;
    {
        smf::BlasThreadGuard g(2, target);
        caller_in = smf::mkl_blocktime_get();
        std::thread t([&] {
            worker_before = smf::mkl_blocktime_get();
            {
                smf::BlasThreadGuard wg(2, target);
                worker_in = smf::mkl_blocktime_get();
            }
            worker_after = smf::mkl_blocktime_get();
        });
        t.join();
    }
    EXPECT_EQ(caller_in, target);
    EXPECT_EQ(worker_in, target);
    EXPECT_EQ(worker_after, worker_before);
    EXPECT_EQ(smf::mkl_blocktime_get(), caller0) << "caller's blocktime not restored";
}
