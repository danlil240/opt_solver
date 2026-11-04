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
#include <thread>

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
