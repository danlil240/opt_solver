#include "smf/multifrontal_serial.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/determinism.hpp"
#include "smf/ldlt_aptp.hpp"
#include "smf/ldlt_kernel.hpp"
#include "smf/solve_pool.hpp"
#include "parallel/chunk_arena.hpp"
#include "parallel/tile_tasks.hpp"
#include "parallel/thread_plan.hpp"
#include "parallel/top_scheduler.hpp"

#include <algorithm>
#include <limits>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <span>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace smf {

namespace {
// Blocks taken from the root arenas by every factorization in the process
// (root_arena_blocks_taken).
std::atomic<long long> g_root_arena_blocks{0};
// Tests (set_late_phase1_thread_for_test): the pool thread whose subtree
// phase starts late, -1 for none; and the dealt tasks other threads took.
std::atomic<int> g_late_phase1_thread{-1};
std::atomic<long long> g_dealt_tasks_taken_over{0};
} // namespace

// Per-thread output and scratch of the tree-parallel path, kept in the
// NumericWorkspace so refactorizations reuse it.
struct TreeParallelBuffers {
  // Reusable, uninitialised contribution-block buffers for the top of the
  // tree: a fresh zero-filled multi-MB allocation per front cost more than
  // the front itself (page faults), so a parent hands its children's
  // buffers back as soon as it has assembled them.
  struct BlockPool {
    std::mutex mtx;
    std::vector<std::pair<std::size_t, std::unique_ptr<double[]>>> free;
    std::unique_ptr<double[]> acquire(std::size_t need, std::size_t &cap) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        std::size_t best = free.size();
        for (std::size_t i = 0; i < free.size(); ++i)
          if (free[i].first >= need && (best == free.size() || free[i].first < free[best].first))
            best = i;
        if (best < free.size()) {
          cap = free[best].first;
          auto buf = std::move(free[best].second);
          free.erase(free.begin() + static_cast<std::ptrdiff_t>(best));
          return buf;
        }
      }
      cap = need;
      return std::unique_ptr<double[]>(new double[need]);
    }
    void release(std::unique_ptr<double[]> buf, std::size_t cap) {
      if (!buf) return;
      std::lock_guard<std::mutex> lk(mtx);
      free.emplace_back(cap, std::move(buf));
    }
  };
  BlockPool pool;
  // Subtree roots' contribution blocks (kRootBlockMinEntries): values and
  // row ids, reset by each factorization's phase 1.
  ChunkArena<double> root_values;
  ChunkArena<Int> root_rows;
  // EarlyGate on the subtree engine: each supernode's reported front work
  // (-1: not yet), reset per armed factorization; grown, never shrunk.
  std::unique_ptr<std::atomic<long long>[]> early_works;
  std::size_t early_works_cap = 0;
  struct Thread {
    std::vector<double> fv, cbv, work;
    std::vector<Int> sri, cbr, pos, rows, cmap;
    std::vector<int8_t> tags;
    std::vector<SolveStep> steps;
    std::vector<DiagSolveEntry> diag;
    TileWorkspace tiles;
  };
  std::vector<Thread> threads;
  // Deterministic mode, serial path: tile-task scratch of the main pass.
  TileWorkspace serial_tiles;
  // The last partition of the tree (partition_subtrees with a delay budget,
  // else partition_tasks) with what it depends on: the analysis, the
  // thread count and which of the two. Refactorizations of an analysis
  // reuse it instead of walking the tree again.
  struct Partition {
    // Kept between calls (and across refactorizations): O(ns) memory.
    std::uint64_t analysis = 0;  // AnalysisKeep::serial; 0: none
    std::size_t ns = 0;
    int threads = 0;
    bool budget = false;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::vector<std::uint8_t> top;
  };
  Partition partition;
  // The tree walks of the thread plan and the partition (tree_view's
  // arrays, subtree_sums), refilled in place for each new analysis: fresh
  // O(ns) buffers page-faulted on every cold call (about 1 ms of plan and
  // partition on boyd2, ns ~ 93k). Both share one view and one set of sums.
  std::vector<Int> view_parent;
  std::vector<double> view_work;
  std::vector<std::uint8_t> view_split;
  SubtreeSums sums;
};

namespace {

// One pass over a sequence of fronts: where it writes and its scratch. The
// serial path is a single pass bound to the FactorKeep and the workspace;
// the tree-parallel path runs one pass per thread over whole subtrees, then
// the serial pass over the top of the tree.
struct FrontPass {
  std::vector<double> *fv;   // factor values; fronts are assembled in place here
  std::vector<Int> *sri;     // solve row indices
  std::vector<SolveStep> *steps;
  std::vector<DiagSolveEntry> *diag;
  // Tree-parallel only: (front, end offset in *diag) after each front this
  // pass factored, so the passes' D entries can be merged in elimination order.
  std::vector<std::pair<Int, std::size_t>> diag_marks;
  std::vector<double> *cbv;  // contribution stack: packed lower blocks
  std::vector<Int> *cbr;     // contribution stack: row ids
  std::vector<Int> *rows;
  std::vector<Int> *cmap;
  std::vector<int8_t> *tags;
  std::vector<double> *work;
  Int *pos;                  // row -> front position, -1 between fronts
  TileWorkspace *tiles = nullptr; // tile-task scratch (top of the tree only)
  std::size_t fo = 0, ro = 0, cv_top = 0, cr_top = 0;
  SerialFactorStats st;
  int64_t growths = 0;
  // Small-front buffers (Control::small_front_kernels): a front of at most
  // kSmallFrontMax rows is assembled and factored in one of these, in the
  // small-front layout (ldlt_factor_small_front), and only its L columns are
  // copied to the factor. When its parent is the next front this pass
  // processes, its contribution block stays in the buffer (node = the front)
  // and the parent assembles straight from it, with no stack copy. The
  // parent itself uses the other buffer, so a delay-budget retry can
  // assemble it again from the intact child block.
  struct SmallBuf {
    alignas(64) double v[kSmallFrontMax * kSmallFrontMax];
    Int rows[kSmallFrontMax];  // the held block's row ids
    Int node = -1;             // supernode whose contribution block is held, or -1
    int ld = 0;                // leading dimension of v while held
    int off = 0;               // the held block starts at row and column off
  };
  SmallBuf sbuf[2];
};

// A contribution block outside the passes' stacks: a top front's, from the
// pool (given back once its parent read it), or a subtree root's, in the
// root arenas (values null, rows unused).
struct NodeBlock {
  std::unique_ptr<double[]> values;
  std::size_t cap = 0;
  std::vector<Int> rows;
  const double *v = nullptr;  // the block's values
  const Int *r = nullptr;     // and row ids
};

// The conditional early stop of a work-limited factorization
// (EarlyWorkStop, Control::scale_on_delay_early_ratio), decided over the
// fronts in postorder whatever order they run in. Each front reports its
// work w before it runs (reach); a cursor walks the postorder over the
// works reported so far, summing them (done) and the fronts' delay-free
// work (free_done, the front included); at the front where done <= at <
// done + w it decides once: stop iff (done + w) / free_done > proj. The
// works are those of the finished factorization, so the decision is a
// function of the factor alone: the same at every thread count in
// deterministic mode. The serial engine reaches the fronts in postorder
// (the cursor is the front itself). On the subtree engine the threads
// store their fronts' works (`works`, per supernode) and add them to a
// shared total; while the total is at most `at` no postorder prefix can
// have crossed (a prefix's sum is at most the total reported), so the
// report returns without the lock. The last front of the crossing prefix
// to report finds the total past `at` and walks the cursor under the lock
// (the acq_rel adds publish the works stored before them), so a reached
// crossing is always decided. The front whose report makes it stop is not
// factored (any front, on the subtree engine). At > 1 thread a crossing
// inside a phase-1 subtree that comes after a top front in postorder is
// decided only once phase 2 has reported that top front.
class EarlyGate {
 public:
  // `works`: the subtree engine's per-supernode slots (null: serial).
  EarlyGate(const AnalysisKeep &keep, const EarlyWorkStop &e, long long at,
            std::atomic<long long> *works)
      : keep_(keep), at_(at), proj_(e.proj), works_(works) {
    if (works_ != nullptr)
      for (std::size_t s = 0; s < keep.supernodes.size(); ++s)
        works_[s].store(-1, std::memory_order_relaxed);
  }

  // Supernode s, about to run with work w: true = stop the factorization.
  bool reach(Int s, long long w) {
    if (verdict_.load(std::memory_order_acquire) != kOpen) return stopped();
    if (works_ == nullptr) {  // the serial engine: s is the cursor's front
      assert(keep_.postorder[cursor_] == s);
      step(w);
      return stopped();
    }
    works_[static_cast<std::size_t>(s)].store(w, std::memory_order_relaxed);
    if (total_.fetch_add(w, std::memory_order_acq_rel) + w <= at_) return false;
    std::lock_guard<std::mutex> lock(mu_);
    const std::size_t ns = keep_.postorder.size();
    while (verdict_.load(std::memory_order_relaxed) == kOpen && cursor_ < ns) {
      const long long wk =
          works_[static_cast<std::size_t>(keep_.postorder[cursor_])].load(std::memory_order_relaxed);
      if (wk < 0) break;
      step(wk);
    }
    return stopped();
  }

 private:
  static constexpr int kOpen = 0, kGo = 1, kStop = 2;
  bool stopped() const { return verdict_.load(std::memory_order_acquire) == kStop; }
  // The front at the cursor, with work w.
  void step(long long w) {
    const Int s = keep_.postorder[cursor_];
    const double f = static_cast<double>(keep_.front_size(s));
    free_done_ += static_cast<double>(keep_.supernodes[static_cast<std::size_t>(s)].width()) * f * f;
    if (done_ <= at_ && done_ + w > at_) {
      const bool stop = static_cast<double>(done_ + w) > proj_ * free_done_;
      verdict_.store(stop ? kStop : kGo, std::memory_order_release);
    }
    done_ += w;
    ++cursor_;
  }

  const AnalysisKeep &keep_;
  long long at_;
  double proj_;
  std::atomic<long long> *works_;     // subtree engine: reported works, -1 none yet
  std::atomic<long long> total_{0};   // subtree engine: sum of the reported works
  std::mutex mu_;
  std::size_t cursor_ = 0;  // postorder index of the next front to account
  long long done_ = 0;
  double free_done_ = 0.0;
  std::atomic<int> verdict_{kOpen};
};

struct FactorContext {
  const AnalysisKeep &keep;
  const Int *acp;
  const Int *ari;
  const double *av;
  const LdltKernelParams &prm;
  bool posdef;
  FactorKeep &fkeep;
  NumericWorkspace &ws;
  FrontPass *const *passes; // indexed by ws.cb_owner (>= 0)
  int *front_pass;          // per supernode: the pass that factored it
  // Contribution blocks kept per node (ws.cb_owner == -1): used where
  // stacks are not popped in order, so that no reader's block can move.
  std::vector<NodeBlock> *node_blocks;
  TreeParallelBuffers::BlockPool *pool;
  // Fallback static pivot (absolute; 0 = off): replaces prm.static_pivot in
  // a root front and in a front receiving more delays than its width.
  double fallback_static = 0.0;
  // APTP front kernel (Control::front_kernel): null = off. Used on fronts
  // that pivot with prm itself (not a fallback static front) and have at
  // least aptp_min_elig fully summed columns.
  const AptpParams *aprm = nullptr;
  int aptp_min_elig = 0;
  // Control::delay_budget (-1 = off) and the absolute static pivot of the
  // fronts that run after the budget is spent.
  int delay_budget = -1;
  double budget_static = 0.0;
  // Control::top_tiles: APTP parameters of the 2D tile tasks (null = off),
  // used on plain threshold fronts of >= kTopTileMinFront rows processed
  // with a TileCtx (the top of the tree).
  const AptpParams *tile_aprm = nullptr;
  // Control::small_front_kernels: FrontPass::sbuf path for small fronts.
  bool small_fronts = false;
  // Work limit (scale on delay; -1 = off): a front whose own work would take
  // the count past it is not factored. work_shared: the threads' shared
  // count (null: the pass's own, the serial engine).
  long long work_cap = -1;
  const std::atomic<long long> *work_shared = nullptr;
  // Tree-parallel: the top-part flags per supernode (null: the serial
  // engine), and the arenas where a phase-1 front whose parent is a top
  // front puts a contribution block of at least kRootBlockMinEntries.
  const std::uint8_t *top = nullptr;
  ChunkArena<double> *root_values = nullptr;
  ChunkArena<Int> *root_rows = nullptr;
  // The conditional early stop under the work limit (null: off).
  EarlyGate *early = nullptr;
};

// The contribution block of a subtree root (its parent is in the top part)
// lives until the top phase reads it. On its pass's stack it would stay
// under every later task's fronts, so a pass's stack would grow to the sum
// of the root blocks of the tasks its thread took: on TSOPF_FS_b300_c3,
// 6203 roots, 1.9 GB in all, 280-450 MB per pass, and a pass grown again
// (a reallocation with a copy, then fresh pages) whenever a factorization
// deals the tasks differently, which stalled phase 1 by 30-50 ms. A root
// block of at least this many entries (32 KB) goes to the root arenas
// instead, whose size is the factorization's total whatever the deal;
// smaller ones stay on the stack (an arena block costs a lock, and trees
// of tens of thousands of tiny subtrees exist).
constexpr std::size_t kRootBlockMinEntries = std::size_t{1} << 12;

// Threads of a front at the top of the tree: its share of the BLAS-thread
// budget, and the pool's idle threads to recruit for its tile tasks.
struct TileCtx {
  int nthreads = 1;
  TileHelpers *helpers = nullptr;
};

// Running delay budget state (Control::delay_budget) of a sequence of
// fronts: the delays counted so far and whether the budget is spent (the
// count reached it, or a front would have taken it past it). Once spent,
// every further front of the sequence static-pivots.
struct BudgetState {
  int count = 0;
  bool spent = false;
};

// Written one past the APTP kernel's declared workspace and checked after
// it: a kernel that writes beyond aptp_work_size() fails the factorization
// with ErrorCode::InternalError (and an assert in debug builds).
constexpr double kWorkCanary = -1.2345678901234567e-300;

// dst[0, n) = src[0, n) for the short columns of small fronts, inline: a
// memmove call per column costs more than the copy.
inline void copy_short(const double *src, double *dst, std::size_t n) noexcept {
  std::size_t i = 0;
#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
#endif
  for (; i < n; ++i) dst[i] = src[i];
}

// Kernel workspace (doubles) of a front: ldlt_factor_front's, or with APTP
// the larger of APTP's plus its canary and the TPP fallback's.
std::size_t front_work_need(int f, int num_elig, const LdltKernelParams &prm,
                            const AptpParams *aprm) {
  const std::size_t tpp = ldlt_work_size(f, prm);
  if (aprm == nullptr) return tpp;
  return std::max(tpp, aptp_work_size(f, num_elig, *aprm) + 1);
}

// Size-as-high-water-mark growth: only the excess beyond the previous size
// is value-initialised, and capacity grows geometrically.
template <class Vec> void grow(Vec &vec, std::size_t need, int64_t &growths) {
  if (vec.size() >= need) return;
  if (vec.capacity() < need) {
    ++growths;
    vec.reserve(std::max(need, vec.capacity() + vec.capacity() / 2));
  }
  vec.resize(need);
}

// Assemble, factor and store supernode s with pass P (index `me` in
// cx.passes). Children's contribution blocks are read from whichever pass
// produced them; with `pop` (postorder passes, where they are the top of the
// stack) blocks on P's own stack are popped. `bs` (null: no budget) is the
// delay budget state of the sequence s belongs to, updated by this front.
// `next` is the front P processes right after s (-1: none or unknown): a
// small front whose parent it is hands its contribution block over in
// P.sbuf.
ErrorCode process_front(const FactorContext &cx, FrontPass &P, int me, Int s, bool pop,
                        BudgetState *bs, const TileCtx *tc = nullptr, Int next = -1) {
  const AnalysisKeep &keep = cx.keep;
  NumericWorkspace &ws = cx.ws;
  FactorKeep &fkeep = cx.fkeep;
  const std::size_t si = static_cast<std::size_t>(s);
  const Supernode &sn = keep.supernodes[si];
  const std::span<const Int> frows = keep.front_rows(s);
  const std::span<const Int> kids = keep.children(s);
  const int p_sym = static_cast<int>(sn.width());
  const int f_sym = static_cast<int>(frows.size());
  Int *const pos = P.pos;
  cx.front_pass[si] = me;

  // A child's contribution block held in one of P's small-front buffers
  // (the previous front's): that buffer, or -1.
  const auto held_buf = [&](std::size_t ci) -> int {
    const Int c = static_cast<Int>(ci);
    return P.sbuf[0].node == c ? 0 : (P.sbuf[1].node == c ? 1 : -1);
  };
  const auto child_rows = [&](std::size_t ci) -> const Int * {
    if (const int b = held_buf(ci); b >= 0) return P.sbuf[b].rows;
    if (ws.cb_owner[ci] < 0) return (*cx.node_blocks)[ci].r;
    return cx.passes[ws.cb_owner[ci]]->cbr->data() + ws.cb_row_off[ci];
  };
  // Packed lower block, or (held in a small-front buffer) its first column
  // with leading dimension ld > 0.
  const auto child_values = [&](std::size_t ci, std::size_t &ld) -> const double * {
    ld = 0;
    if (const int b = held_buf(ci); b >= 0) {
      const FrontPass::SmallBuf &h = P.sbuf[b];
      ld = static_cast<std::size_t>(h.ld);
      return h.v + static_cast<std::size_t>(h.off) * (ld + 1);
    }
    if (ws.cb_owner[ci] < 0) return (*cx.node_blocks)[ci].v;
    return cx.passes[ws.cb_owner[ci]]->cbv->data() + ws.cb_value_off[ci];
  };

  // Front rows: delayed columns from children, own pivots, then the
  // symbolic extension rows.
  int nd = 0;
  for (const Int c : kids) nd += ws.cb_delayed[static_cast<std::size_t>(c)];
  grow(*P.rows, static_cast<std::size_t>(nd + f_sym), P.growths);
  Int *rows = P.rows->data();
  int f = 0;
  if (nd > 0) {
    for (const Int c : kids) {
      const std::size_t ci = static_cast<std::size_t>(c);
      const Int *cr = child_rows(ci);
      for (Int t = 0; t < ws.cb_delayed[ci]; ++t) {
        rows[f] = cr[t];
        pos[cr[t]] = f++;
      }
    }
  }
  for (int t = 0; t < f_sym; ++t) {
    const Int r = frows[static_cast<std::size_t>(t)];
    rows[f] = r;
    pos[r] = f++;
  }
  const int num_elig = nd + p_sym;
  // Defensive: a child row outside the symbolic structure joins the front.
  for (const Int c : kids) {
    const std::size_t ci = static_cast<std::size_t>(c);
    const Int *cr = child_rows(ci);
    for (Int t = ws.cb_delayed[ci]; t < ws.cb_size[ci]; ++t) {
      if (pos[cr[t]] < 0) {
        grow(*P.rows, static_cast<std::size_t>(f) + 1, P.growths);
        rows = P.rows->data();
        rows[f] = cr[t];
        pos[cr[t]] = f++;
      }
    }
  }

  const std::size_t sf = static_cast<std::size_t>(f);
  // The front is assembled in place at its factor offset: after the
  // kernel, columns [0, p) are already L in the solve layout, and the
  // trailing block is dead once pushed to the stack (the next front
  // starts at fo + f * p and reuses it).
  grow(*P.fv, P.fo + sf * sf, P.growths);
  grow(*P.cmap, sf, P.growths);
  grow(*P.tags, sf, P.growths);
  // Delay budget (Control::delay_budget): once spent, the front
  // static-pivots like a static_pivot_tol front.
  const bool budget_on = bs != nullptr && cx.delay_budget >= 0 && cx.budget_static > 0.0;
  if (budget_on && bs->count >= cx.delay_budget) bs->spent = true;
  bool budget_front = budget_on && bs->spent;
  // APTP only on plain threshold-pivoting fronts: a fallback or budget
  // static front (below) keeps the TPP kernel.
  const bool fallback_front = cx.fallback_static > 0.0 && (sn.parent == -1 || nd > p_sym);
  // 2D tile tasks (top of the tree): APTP block structure whatever the
  // front kernel, on the same plain threshold fronts.
  bool tiled = tc != nullptr && P.tiles != nullptr && tiles_front(cx.tile_aprm != nullptr, f) &&
               !fallback_front && !budget_front;
  const AptpParams *aprm = tiled ? cx.tile_aprm
                           : (cx.aprm != nullptr && !fallback_front && !budget_front &&
                              num_elig >= cx.aptp_min_elig)
                               ? cx.aprm
                               : nullptr;
  grow(*P.work, front_work_need(f, num_elig, cx.prm, aprm), P.growths);
  double *F = P.fv->data() + P.fo;
  // Small fronts (Control::small_front_kernels; never APTP or tiled): the
  // front is assembled and factored in the small-front buffer not holding a
  // child's block, Fa with leading dimension lda; otherwise Fa = F, lda = f.
  // Only where ldlt_factor_front would take its unblocked path, whose factor
  // ldlt_factor_small_front reproduces bit for bit.
  const bool small = cx.small_fronts && f <= kSmallFrontMax && f < cx.prm.blocked_min_front &&
                     cx.prm.nb > 1 && aprm == nullptr;
  int sb = -1;
  double *Fa = F;
  std::size_t lda = sf;
  if (small) {
    sb = P.sbuf[0].node >= 0 ? 1 : 0;
    assert(P.sbuf[sb].node < 0);
    Fa = P.sbuf[sb].v;
    lda = static_cast<std::size_t>(small_front_ld(f));
  }

  // Fallback static pivoting: only where delaying fails. A root has no
  // parent to delay to (its unaccepted columns would be zero pivots), and a
  // front that receives more delayed columns than its own width is over its
  // delay budget. Every other front pivots with delays as usual. A pivot
  // forced there from an exact zero stays a zero eigenvalue (hybrid
  // accounting), so a singular matrix is still reported singular; one raised
  // from a nonzero value keeps its sign unless ctrl.static_pivots_as_zero.
  // A fallback front keeps its own tolerance and hybrid accounting; a budget
  // front is a static_pivot_tol front (the Control's accounting).
  const LdltKernelParams *prm = &cx.prm;
  LdltKernelParams fallback_prm;
  LdltKernelParams budget_prm;
  if (budget_on) {
    budget_prm = cx.prm;
    budget_prm.static_pivot = cx.budget_static;
  }
  if (fallback_front) {
    fallback_prm = cx.prm;
    fallback_prm.static_pivot = cx.fallback_static;
    fallback_prm.static_zero_stays_zero = true;
    if (fallback_prm.min_abs_pivot > 0.0) fallback_prm.min_abs_pivot = cx.fallback_static;
    prm = &fallback_prm;
  } else if (budget_front) {
    prm = &budget_prm;
  }

  int8_t *tags = P.tags->data();
  SerialFactorStats &st = P.st;
  double *const work = P.work->data();
  const bool is_root = (sn.parent == -1);
  std::size_t new_cv_top = P.cv_top;
  std::size_t new_cr_top = P.cr_top;
  Int *cm = P.cmap->data();
  // Without `pop`, children's blocks live in node buffers: back to the pool
  // once this front no longer reads them (after its kernel, or on failure).
  const auto release_child_blocks = [&] {
    if (pop) return;
    for (const Int c : kids) {
      const std::size_t ci = static_cast<std::size_t>(c);
      if (ws.cb_size[ci] > 0 && ws.cb_owner[ci] < 0) {
        NodeBlock &nb = (*cx.node_blocks)[ci];
        cx.pool->release(std::move(nb.values), nb.cap);
      }
    }
  };
  // A front's work (eligible columns x rows^2, as counted below) is known
  // before its kernel runs. Work is only ever added, so a front that would
  // take the count past the limit means the finished run exceeds it too:
  // stop before this front instead of after it (the same decision; a
  // delay cascade's last front can be most of the work). Its work is
  // counted, so a stopped run reports front work past the limit either way.
  // The early gate (EarlyGate) stops it the same way.
  if (cx.work_cap >= 0) {
    const long long w = static_cast<long long>(num_elig) * f * f;
    const long long done = cx.work_shared != nullptr
                               ? cx.work_shared->load(std::memory_order_relaxed)
                               : P.st.front_work;
    if (done + w > cx.work_cap || (cx.early != nullptr && cx.early->reach(s, w))) {
      P.st.front_work += w;
      for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
      release_child_blocks();
      return ErrorCode::WorkLimitExceeded;
    }
  }
  LdltKernelResult kr;
  int aptp_failed = 0;
  const int64_t tile_growths0 = P.tiles != nullptr ? P.tiles->growths : 0;
  // Assemble and factor; at most twice. Children's contribution blocks (the
  // top of P's stack when P made them) are only read here, and nothing is
  // pushed or released before this front's own Schur complement, so a delay
  // budget retry can assemble the front again.
  for (;;) {
    if (small) {
      // The small-front layout: rows [f, fe) zero, and the rows above the
      // diagonal its kernels write (down to the 4-row boundary) finite.
      const std::size_t fe = (sf + 3) & ~std::size_t{3};
      for (std::size_t j = 0; j < sf; ++j) {
        const std::size_t r0 = j & ~std::size_t{3};
        std::fill_n(Fa + j * lda + r0, fe - r0, 0.0);
      }
    } else {
      for (std::size_t j = 0; j < sf; ++j) {
        std::fill_n(F + j * sf + j, sf - j, 0.0);
      }
    }

    // Original entries of the pivot columns (lower CSC, so pos[row] >= pos[col]).
    for (Int j = sn.col_start; j < sn.col_end; ++j) {
      double *Fj = Fa + static_cast<std::size_t>(pos[j]) * lda;
      for (Int k = cx.acp[j]; k < cx.acp[j + 1]; ++k) {
        Fj[pos[cx.ari[k]]] += cx.av[k];
      }
    }

    for (const Int c : kids) {
      const std::size_t ci = static_cast<std::size_t>(c);
      const Int q = ws.cb_size[ci];
      if (q <= 0) continue;
      const Int *cr = child_rows(ci);
      std::size_t cld = 0;  // > 0: strided block in a small-front buffer
      const double *const cv0 = child_values(ci, cld);
      const double *cv = cv0;
      bool monotone = true;
      Int prev = -1;
      for (Int t = 0; t < q; ++t) {
        const Int l = pos[cr[t]];
        cm[t] = l;
        monotone = monotone && (l > prev);
        prev = l;
      }
      if (monotone) {
        for (Int jj = 0; jj < q; ++jj) {
          if (cld > 0) cv = cv0 + static_cast<std::size_t>(jj) * (cld + 1);
          double *Fc = Fa + static_cast<std::size_t>(cm[jj]) * lda;
          for (Int ii = jj; ii < q; ++ii) Fc[cm[ii]] += *cv++;
        }
      } else {
        for (Int jj = 0; jj < q; ++jj) {
          if (cld > 0) cv = cv0 + static_cast<std::size_t>(jj) * (cld + 1);
          const std::size_t lj = static_cast<std::size_t>(cm[jj]);
          for (Int ii = jj; ii < q; ++ii) {
            const std::size_t li = static_cast<std::size_t>(cm[ii]);
            if (li >= lj) Fa[lj * lda + li] += *cv++;
            else Fa[li * lda + lj] += *cv++;
          }
        }
      }
      if (pop && ws.cb_owner[ci] == me) {
        new_cv_top = std::min(new_cv_top, static_cast<std::size_t>(ws.cb_value_off[ci]));
        new_cr_top = std::min(new_cr_top, static_cast<std::size_t>(ws.cb_row_off[ci]));
      }
    }

    kr = LdltKernelResult{};
    aptp_failed = 0;
    if (aprm != nullptr) {
      // APTP, then the blocked TPP kernel on the columns it failed (their
      // Schur state is current); what TPP fails too is delayed as usual.
      const std::size_t need = aptp_work_size(f, num_elig, *aprm);
      work[need] = kWorkCanary;
      const AptpResult ar =
          tiled ? ldlt_aptp_front_tiled(F, f, num_elig, rows, *aprm, tags, work, *P.tiles,
                                        tc->nthreads, tc->helpers)
                : ldlt_aptp_front(F, f, num_elig, rows, *aprm, tags, work);
      if (work[need] != kWorkCanary) {
        assert(false && "APTP front kernel overran its workspace");
        for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
        release_child_blocks();
        return ErrorCode::InternalError;
      }
      aptp_failed = ar.num_failed;
      if (ar.num_failed > 0) {
        kr = ldlt_factor_front(F, f, num_elig, rows, *prm, tags, work, ar.num_eliminated);
      } else {
        kr.num_accepted = ar.num_eliminated;
      }
      kr.inertia.positive += ar.inertia.positive;
      kr.inertia.negative += ar.inertia.negative;
      kr.inertia.zero += ar.inertia.zero;
      kr.flops += ar.flops;
    } else if (small) {
      kr = ldlt_factor_small_front(Fa, static_cast<int>(lda), f, num_elig, rows, *prm, tags);
    } else {
      kr = ldlt_factor_front(F, f, num_elig, rows, *prm, tags, work);
    }
    // Delay budget: a front whose delays would take the count past the
    // budget is assembled again and static-pivoted (it then delays nothing),
    // and the budget is spent. A root's unaccepted columns are zero pivots,
    // not delays.
    if (!budget_on || budget_front || fallback_front || is_root ||
        bs->count + (num_elig - kr.num_accepted) <= cx.delay_budget)
      break;
    bs->spent = true;
    budget_front = true;
    prm = &budget_prm;
    aprm = nullptr;
    tiled = false;
    ++P.st.budget_retries;
    // The kernels permuted rows[] into pivot order (sym_swap / view_swap)
    // but not pos[], which assembly scatters through: put rows[] back in
    // assembly order, rows[pos[r]] = r, with the child map as scratch.
    for (int t = 0; t < f; ++t) cm[pos[rows[t]]] = rows[t];
    std::copy(cm, cm + f, rows);
  }
  // Children's blocks held in small-front buffers are consumed.
  for (const Int c : kids) {
    if (const int b = held_buf(static_cast<std::size_t>(c)); b >= 0) P.sbuf[b].node = -1;
  }
  // A small front's L columns (with D) go to the factor; its contribution
  // block is read from the buffer below.
  if (small) {
    for (int c = 0; c < kr.num_accepted; ++c) {
      const std::size_t uc = static_cast<std::size_t>(c);
      copy_short(Fa + uc * lda + uc, F + uc * sf + uc, sf - uc);
    }
  }
  st.aptp_failed += aptp_failed;
  if (tiled) ++st.top_tile_fronts;
  if (P.tiles != nullptr) P.growths += P.tiles->growths - tile_growths0;
  const int p = kr.num_accepted;
  if (budget_on && !is_root) bs->count += num_elig - p;
  P.cv_top = new_cv_top;
  P.cr_top = new_cr_top;
  release_child_blocks();
  st.inertia.positive += kr.inertia.positive;
  st.inertia.negative += kr.inertia.negative;
  st.inertia.zero += kr.inertia.zero;
  st.static_pivots += kr.static_pivots;
  st.flops += kr.flops;
  st.front_work += static_cast<long long>(num_elig) * f * f;
  st.max_front = std::max(st.max_front, f);

  if (cx.posdef && p < num_elig) {
    for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
    return ErrorCode::NotPositiveDefinite;
  }

  // A root has no parent to delay to: unaccepted columns are zero pivots
  // (marked was_zero below). With a static pivot the kernel accepts every
  // column, so a static pivot is never one of them: it is a real 1x1 pivot
  // (tag 1, d raised) that the solve divides by and realized_pivot_order
  // keeps in place.
  int p_store = p;
  if (is_root && p < num_elig) {
    for (int c = p; c < num_elig; ++c) tags[c] = 0;
    st.inertia.zero += num_elig - p;
    p_store = num_elig;
  }

  // Overflow anywhere in a front propagates into a later pivot of the same
  // or an ancestor front, so checking every D entry catches it without a
  // per-entry test elsewhere. Only the pivot blocks are rewritten: D is
  // read out first, then the strict upper part is zeroed and the unit (or,
  // for SPD, sqrt(d)) diagonal written.
  bool finite_d = true;
  if (cx.posdef) {
    for (int c = 0; c < p_store; ++c) {
      double *Lc = F + static_cast<std::size_t>(c) * sf;
      const double sd = std::sqrt(Lc[c]);
      std::fill_n(Lc, c, 0.0);
      Lc[c] = sd;
      for (int i = c + 1; i < f; ++i) Lc[i] *= sd;
    }
  } else {
    // L is unit lower, so its diagonal slots keep D (FactorKeep::inline_d):
    // d on the diagonal, a 2x2 block's off-diagonal in the strictly upper
    // slot (c, c + 1), which no triangular kernel reads.
    double d10 = 0.0;
    for (int c = 0; c < p_store; ++c) {
      double *Lc = F + static_cast<std::size_t>(c) * sf;
      const int8_t tag = tags[c];
      if (tag == 0) {
        P.diag->push_back(DiagSolveEntry{1, 1, rows[c], 0, 0.0, 0.0, 0.0});
        std::fill_n(Lc + c + 1, f - c - 1, 0.0);
        Lc[c] = 0.0;
      } else if (tag == 2) {
        P.diag->push_back(DiagSolveEntry{
            2, 0, rows[c], rows[c + 1], Lc[c], Lc[c + 1], Lc[sf + c + 1]});
        finite_d = finite_d && is_finite_fast(Lc[c]) && is_finite_fast(Lc[c + 1]) &&
                   is_finite_fast(Lc[sf + c + 1]);
        d10 = Lc[c + 1];
        Lc[c + 1] = 0.0;
      } else if (tag == 1) {
        P.diag->push_back(DiagSolveEntry{1, 0, rows[c], 0, Lc[c], 0.0, 0.0});
        finite_d = finite_d && is_finite_fast(Lc[c]);
      }
      std::fill_n(Lc, c, 0.0);
      if (tag == -1) Lc[c - 1] = d10;
    }
    fkeep.pivot_types[si].assign(tags, tags + p_store);
  }
  fkeep.factor_col_ptr[si] = static_cast<LongInt>(P.fo);
  SolveStep step;
  step.row_offset = static_cast<LongInt>(P.ro);
  step.factor_offset = static_cast<LongInt>(P.fo);
  step.p = p_store;
  step.f = f;
  P.steps->push_back(step);
  grow(*P.sri, P.ro + sf, P.growths);
  std::copy(rows, rows + f, P.sri->begin() + static_cast<std::ptrdiff_t>(P.ro));
  P.fo += sf * static_cast<std::size_t>(p_store);
  P.ro += sf;

  // Push the Schur complement (delayed columns first) for the parent: on
  // P's stack, or into its own node buffer when stacks are not popped.
  const int q = f - p;
  if (!is_root && q > 0 && small && pop && next == sn.parent) {
    // The parent is next: it assembles the block from the buffer.
    FrontPass::SmallBuf &h = P.sbuf[sb];
    h.node = s;
    h.ld = static_cast<int>(lda);
    h.off = p;
    std::copy(rows + p, rows + f, h.rows);
    ws.cb_owner[si] = me;
    ws.cb_value_off[si] = static_cast<LongInt>(P.cv_top);
    ws.cb_row_off[si] = static_cast<LongInt>(P.cr_top);
    ws.cb_size[si] = q;
    ws.cb_delayed[si] = num_elig - p;
    st.delayed += num_elig - p;
    ++st.small_front_handovers;
  } else if (!is_root && q > 0) {
    const std::size_t sq = static_cast<std::size_t>(q);
    double *dst;
    Int *dst_rows;
    const bool root_block = pop && cx.top != nullptr &&
                            cx.top[static_cast<std::size_t>(sn.parent)] != 0 &&
                            sq * (sq + 1) / 2 >= kRootBlockMinEntries;
    if (pop && !root_block) {
      grow(*P.cbv, P.cv_top + sq * (sq + 1) / 2, P.growths);
      grow(*P.cbr, P.cr_top + sq, P.growths);
      dst = P.cbv->data() + P.cv_top;
      dst_rows = P.cbr->data() + P.cr_top;
      ws.cb_owner[si] = me;
      ws.cb_value_off[si] = static_cast<LongInt>(P.cv_top);
      ws.cb_row_off[si] = static_cast<LongInt>(P.cr_top);
      P.cv_top += sq * (sq + 1) / 2;
      P.cr_top += sq;
    } else {
      NodeBlock &nb = (*cx.node_blocks)[si];
      if (root_block) {
        dst = cx.root_values->take(sq * (sq + 1) / 2);
        dst_rows = cx.root_rows->take(sq);
        g_root_arena_blocks.fetch_add(1, std::memory_order_relaxed);
      } else {
        nb.values = cx.pool->acquire(sq * (sq + 1) / 2, nb.cap);
        nb.rows.resize(sq);
        dst = nb.values.get();
        dst_rows = nb.rows.data();
      }
      nb.v = dst;
      nb.r = dst_rows;
      ws.cb_owner[si] = -1;
      ws.cb_value_off[si] = 0;
      ws.cb_row_off[si] = 0;
    }
    for (int jj = 0; jj < q; ++jj) {
      const double *src = Fa + static_cast<std::size_t>(p + jj) * lda + (p + jj);
      if (small) copy_short(src, dst, static_cast<std::size_t>(q - jj));
      else std::copy(src, src + (q - jj), dst);
      dst += q - jj;
    }
    std::copy(rows + p, rows + f, dst_rows);
    ws.cb_size[si] = q;
    ws.cb_delayed[si] = num_elig - p;
    st.delayed += num_elig - p;
  } else {
    ws.cb_size[si] = 0;
    ws.cb_delayed[si] = 0;
  }

  for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
  return finite_d ? ErrorCode::Success : ErrorCode::IllegalValue;
}

void add_stats(SerialFactorStats &to, const SerialFactorStats &from) {
  to.inertia.positive += from.inertia.positive;
  to.inertia.negative += from.inertia.negative;
  to.inertia.zero += from.inertia.zero;
  to.delayed += from.delayed;
  to.static_pivots += from.static_pivots;
  to.aptp_failed += from.aptp_failed;
  to.budget_retries += from.budget_retries;
  to.top_tile_fronts += from.top_tile_fronts;
  to.top_part_fronts += from.top_part_fronts;
  to.small_front_handovers += from.small_front_handovers;
  to.flops += from.flops;
  to.front_work += from.front_work;
  to.max_front = std::max(to.max_front, from.max_front);
}

// Predicted flops of supernode si's front, the term Info::predicted_flops
// sums (assembly_tree.cpp): p f^2 for the symbolic pivot count and size.
double predicted_front_flops(const AnalysisKeep &keep, std::size_t si) {
  const double f = static_cast<double>(keep.front_size(static_cast<Int>(si)));
  const double p = static_cast<double>(keep.supernodes[si].width());
  return p * f * f;
}

// Control::delay_budget counts delays per region of a tree partition made
// with this nominal thread count, whatever Control::num_threads is, so every
// thread count makes the same static/threshold choice per front.
constexpr int kBudgetNominalThreads = 16;

// The gather (phase 3) copies a factor of up to kCachedGatherBytes in pieces
// of kGatherPieceBytes: the CRT's memmove of a whole pass streams the copy
// past the cache, so the solve that usually follows read every factor entry
// back from DRAM. TSOPF_FS_b162_c1 (19 MB, cold, 2-16 threads): first solve
// 1.0-1.1 -> 0.5-0.85 ms, mostly ~0.6 (one thread: 0.8-0.9); gather 0.64 ->
// 0.67 ms. A larger factor cannot stay cached until the solve and is copied
// whole: TSOPF_FS_b300_c3 (435 MB, 16 threads) gathers in 15 ms whole, 20 ms
// in pieces, with the same solve time. 32 MB: the smaller last-level cache
// of this machine's two (CCD1; CCD0 has 96 MB).
constexpr std::size_t kCachedGatherBytes = std::size_t{32} << 20;
constexpr std::size_t kGatherPieceBytes = std::size_t{64} << 10;

template <class T>
void gather_copy(const T *src, std::size_t count, T *dst, bool in_pieces) {
  if (!in_pieces) {
    std::copy(src, src + count, dst);
    return;
  }
  constexpr std::size_t piece = kGatherPieceBytes / sizeof(T);
  for (std::size_t k = 0; k < count; k += piece)
    std::copy(src + k, src + std::min(count, k + piece), dst + k);
}

} // namespace

long long root_arena_blocks_taken() { return g_root_arena_blocks.load(std::memory_order_relaxed); }

void set_late_phase1_thread_for_test(int tid) {
  g_late_phase1_thread.store(tid, std::memory_order_relaxed);
}

long long dealt_tasks_taken_over() {
  return g_dealt_tasks_taken_over.load(std::memory_order_relaxed);
}

ErrorCode multifrontal_factor_serial(const AnalysisKeep &keep, const Int *acp,
                                     const Int *ari, const double *av,
                                     const Control &ctrl, double eff_small,
                                     double static_pivot, double fallback_static_pivot,
                                     double budget_static_pivot,
                                     bool posdef, FactorKeep &fkeep,
                                     NumericWorkspace &ws, SerialFactorStats &st,
                                     double work_limit, const EarlyWorkStop &early) {
  const std::size_t n = static_cast<std::size_t>(keep.n);
  const std::size_t ns = keep.supernodes.size();

  // `plan` sizes from the symbolic structure are not counted as growth.
  auto reserve_plan = [](auto &vec, std::size_t need) {
    if (vec.size() < need) vec.resize(need);
  };

  LdltKernelParams prm;
  prm.u = ctrl.pivot_u;
  prm.eff_small = eff_small;
  prm.small_pivot = ctrl.small_pivot;
  prm.nb = std::max(2, ctrl.indef_panel_size);
  prm.posdef = posdef;
  prm.static_pivot = posdef ? 0.0 : static_pivot;
  prm.static_pivots_as_zero = ctrl.static_pivots_as_zero;
  prm.static_zero_stays_zero = ctrl.static_zero_stays_zero;
  // Control::static_pivot_in_order: 1x1 pivots in order, |d| below the
  // static pivot raised (the quasidefinite mode).
  prm.min_abs_pivot = (prm.static_pivot > 0.0 && ctrl.static_pivot_in_order) ? prm.static_pivot : 0.0;
  const double fallback_static = posdef ? 0.0 : fallback_static_pivot;
  // Control::delay_budget: threshold-pivoting indefinite fronts only (with
  // static_pivot_tol every front already static-pivots and never delays).
  const bool budget_armed = !posdef && prm.static_pivot == 0.0 && ctrl.delay_budget >= 0 &&
                            budget_static_pivot > 0.0;
  const int delay_budget = budget_armed ? ctrl.delay_budget : -1;
  const double budget_static = budget_armed ? budget_static_pivot : 0.0;

  // APTP (Control::front_kernel) on plain threshold-pivoting fronts only: it
  // has no static-pivot mode, so SPD, static_pivot_tol and in-order fronts
  // keep the TPP kernel (fallback static fronts are excluded per front).
  AptpParams aprm;
  aprm.u = ctrl.pivot_u;
  aprm.nb = std::max(1, ctrl.aptp_nb);
  aprm.ib = std::max(1, ctrl.aptp_ib);
  aprm.small = eff_small;
  // Auto with aptp_min_elig = 0 (the default) is Tpp.
  const bool kernel_aptp = ctrl.front_kernel == FrontKernel::Aptp ||
                           (ctrl.front_kernel == FrontKernel::Auto && ctrl.aptp_min_elig > 0);
  const bool aptp_on = kernel_aptp && !posdef && prm.static_pivot == 0.0 &&
                       prm.min_abs_pivot == 0.0 && ctrl.pivot_u > 0.0;
  const AptpParams *aptp = aptp_on ? &aprm : nullptr;
  const int aptp_min_elig =
      ctrl.front_kernel == FrontKernel::Aptp ? 1 : std::max(1, ctrl.aptp_min_elig);
  // Control::top_tiles: 2D tile tasks on the large plain threshold fronts at
  // the top of the tree, whatever front_kernel says (APTP is threshold
  // pivoting with an a posteriori check, as stable as TPP).
  //
  // Deterministic mode (determinism.hpp): the tile kernel's result does not
  // depend on its thread count but differs from the untiled kernels', so
  // every such front of >= kTopTileMinFront rows is tiled, wherever it sits
  // in the tree and whatever num_threads is (1 included); BLAS runs on one
  // thread everywhere, and the factor is laid out in postorder.
  //
  // Below pivot u = kWeakPivotRatio (1e-6; IPOPT's u = 1e-8) no front is
  // tiled, at any thread count and in deterministic mode too: the tile
  // (APTP) kernel lacks TPP's weak-1x1 rule, so those fronts keep the TPP
  // kernel, the same one the 1-thread run uses (ruling R42).
  const bool det = deterministic_mode(ctrl);
  if (det) enable_blas_reproducibility();
  AptpParams tile_aprm = aprm;
  tile_aprm.nb = std::max(1, ctrl.top_tile_nb);
  const bool tiles_on = top_tiles_on(ctrl, posdef, prm.static_pivot, prm.min_abs_pivot, det);
  // Deterministic mode: the tile context of the fronts outside the top
  // phase (one thread, no helpers).
  const TileCtx serial_tc{};
  const TileCtx *const det_tc = det && tiles_on ? &serial_tc : nullptr;

  // Early abort once the negative pivots exceed the limit (indefinite only).
  // Pivots are only ever added, so a run whose final count is within the
  // limit never trips it and does exactly the work of an unarmed run.
  const int neg_limit = posdef ? -1 : ctrl.max_negative_pivots;
  const bool armed = neg_limit >= 0;
  // Early stop once the front work exceeds work_limit (the same monotone
  // argument: an unstopped run's final count decides it). The counts are
  // integers, summed exactly in any order.
  const bool work_armed = !posdef && work_limit >= 0.0;
  const long long work_cap =
      !work_armed ? -1
                  : (work_limit >= 9.0e18 ? std::numeric_limits<long long>::max()
                                          : static_cast<long long>(std::floor(work_limit)));

  // Symbolic plan: largest front, factor size, the peak of the
  // contribution stack when no pivot is delayed, and the tree's work (sum of
  // f^2 p, tree_view's unit) for plan_is_serial.
  std::size_t max_f = 1;
  std::size_t max_work = 1;
  std::size_t pred_factor = 0;
  std::size_t pred_rows = 0;
  double tree_work = 0.0;
  ws.cb_value_off.resize(ns);
  ws.cb_row_off.resize(ns);
  ws.cb_size.resize(ns);
  ws.cb_delayed.resize(ns);
  ws.cb_owner.assign(ns, 0);
  {
    std::size_t vtop = 0, rtop = 0, vpeak = 0, rpeak = 0;
    for (const Int s : keep.postorder) {
      const std::size_t si = static_cast<std::size_t>(s);
      const Supernode &sn = keep.supernodes[si];
      const std::size_t f = static_cast<std::size_t>(keep.front_size(static_cast<Int>(si)));
      const std::size_t p = static_cast<std::size_t>(sn.width());
      max_f = std::max(max_f, f);
      max_work = std::max(
          max_work, front_work_need(static_cast<int>(f), static_cast<int>(p), prm,
                                    static_cast<int>(p) >= aptp_min_elig ? aptp : nullptr));
      pred_factor += f * p;
      pred_rows += f;
      const double fd = static_cast<double>(f);
      tree_work += fd * fd * static_cast<double>(p);
      for (const Int c : keep.children(s)) {
        const std::size_t ci = static_cast<std::size_t>(c);
        if (ws.cb_size[ci] > 0) {
          vtop = std::min(vtop, static_cast<std::size_t>(ws.cb_value_off[ci]));
          rtop = std::min(rtop, static_cast<std::size_t>(ws.cb_row_off[ci]));
        }
      }
      const std::size_t q = f - p;
      if (sn.parent >= 0 && q > 0) {
        ws.cb_value_off[si] = static_cast<LongInt>(vtop);
        ws.cb_row_off[si] = static_cast<LongInt>(rtop);
        ws.cb_size[si] = static_cast<Int>(q);
        vtop += q * (q + 1) / 2;
        rtop += q;
        vpeak = std::max(vpeak, vtop);
        rpeak = std::max(rpeak, rtop);
      } else {
        ws.cb_size[si] = 0;
      }
    }
    reserve_plan(ws.cb_values, vpeak);
    reserve_plan(ws.cb_rows, rpeak);
  }

  if (ws.row_pos.size() != n) ws.row_pos.assign(n, -1);
  reserve_plan(ws.front_rows, max_f);
  reserve_plan(ws.child_map, max_f);
  reserve_plan(ws.pivot_tags, max_f);
  reserve_plan(ws.kernel_work, max_work);

  auto &fv = fkeep.factor_values;
  auto &sri = fkeep.solve_row_indices;
  reserve_plan(fv, pred_factor);
  reserve_plan(sri, pred_rows);
  fkeep.factor_col_ptr.resize(ns + 1);
  if (!posdef) fkeep.pivot_types.resize(ns);
  fkeep.solve_steps.clear();
  fkeep.inv_diag_offset.clear();  // new steps: no cached L11^-1 yet
  fkeep.solve_steps.reserve(ns);
  fkeep.solve_subtrees.clear();
  fkeep.solve_threads = 1;
  fkeep.inline_d = !posdef;
  fkeep.solve_top_begin = 0;
  fkeep.solve_top_rows.clear();
  fkeep.solve_top_parent.clear();
  fkeep.diag_entries.clear();
  if (!posdef) fkeep.diag_entries.reserve(n);

  FrontPass main;
  main.fv = &fv;
  main.sri = &sri;
  main.steps = &fkeep.solve_steps;
  main.diag = &fkeep.diag_entries;
  main.cbv = &ws.cb_values;
  main.cbr = &ws.cb_rows;
  main.rows = &ws.front_rows;
  main.cmap = &ws.child_map;
  main.tags = &ws.pivot_tags;
  main.work = &ws.kernel_work;
  main.pos = ws.row_pos.data();

  std::vector<FrontPass *> passes{&main};
  std::vector<int> front_pass(ns, 0);
  std::vector<NodeBlock> node_blocks;
  ErrorCode status = ErrorCode::Success;

  // Tree-parallel when there are threads to use and at least two subtrees.
  // Plain std::threads, not OpenMP: MSVC's OpenMP runtime would be loaded
  // next to the ones MKL / CHOLMOD / OpenBLAS bring, which aborts.
  //
  // With a delay budget the partition is the budget's (kBudgetNominalThreads),
  // for every num_threads including 1: its subtrees are the budget's counting
  // regions, so scheduling uses them as they are.
  //
  // Control::auto_threads: at most the thread count plan_threads picks for
  // the tree (else num_threads); below 2, the serial engine. Deterministic
  // mode's results do not depend on it: every thread count tiles the same
  // fronts there. The plan depends on the tree, num_threads, the costs and
  // the split fronts (tiles_on) alone: ws.thread_plan keeps it for the
  // refactorizations of this analysis.
  //
  // Without a delay budget the subtrees run as partition_tasks' tasks
  // (light adjacent subtrees merged). The partition depends on the tree and
  // its thread count alone: ws.tree_par->partition keeps it.
  //
  // A tree whose work alone rules out every T > 1 (plan_is_serial: below
  // kTreeParallelMinWork, or within the per-thread cost) takes the serial
  // engine without the plan, the tree view or the partition, all of which
  // would come out serial (about 1 ms per analysis on boyd1, ns ~ 90k).
  // The plan trace (SMF_BENCH_PLAN_TRACE) still evaluates the plan.
  int threads = 1;
  const auto buffers = [&ws]() -> TreeParallelBuffers & {
    if (!ws.tree_par) ws.tree_par = std::make_shared<TreeParallelBuffers>();
    return *ws.tree_par;
  };
  TreeView tv;
  bool have_tv = false;
  const auto view = [&]() -> const TreeView & {
    if (!have_tv) {
      TreeParallelBuffers &b = buffers();
      tv = tree_view(keep, tiles_on, b.view_parent, b.view_work, b.view_split);
    }
    have_tv = true;
    return tv;
  };
  bool have_sums = false;
  const auto sums = [&]() -> SubtreeSums & {
    if (!have_sums) subtree_sums(view(), buffers().sums);
    have_sums = true;
    return buffers().sums;
  };
  const ThreadCosts thread_costs{ctrl.thread_cost_per_thread, ctrl.thread_cost_per_top_front,
                                 ctrl.thread_cost_per_split_front_thread,
                                 ctrl.thread_split_scaling_cap};
  int want_threads = ns >= 2 ? ctrl.num_threads : 1;
  const bool below_min_work = tree_work < kTreeParallelMinWork;
  if (ctrl.auto_threads && want_threads > 1 && plan_is_serial(tree_work, thread_costs) &&
      !plan_trace_enabled()) {
    want_threads = 1;
  } else if (ctrl.auto_threads && want_threads > 1) {
    auto &pc = ws.thread_plan;
    if (pc.analysis != keep.serial || pc.num_threads != ctrl.num_threads ||
        pc.per_thread != thread_costs.per_thread ||
        pc.per_top_front != thread_costs.per_top_front ||
        pc.per_split_front_thread != thread_costs.per_split_front_thread ||
        pc.split_cap != thread_costs.split_cap || pc.tiles_on != tiles_on) {
      pc.threads = plan_threads(view(), sums(), ctrl.num_threads, thread_costs).threads;
      pc.analysis = keep.serial;
      pc.num_threads = ctrl.num_threads;
      pc.per_thread = thread_costs.per_thread;
      pc.per_top_front = thread_costs.per_top_front;
      pc.per_split_front_thread = thread_costs.per_split_front_thread;
      pc.split_cap = thread_costs.split_cap;
      pc.tiles_on = tiles_on;
    }
    want_threads = pc.threads;
  } else if (want_threads > 1 && plan_trace_enabled()) {
    // auto_threads off: the plan is evaluated for SMF_BENCH_PLAN_TRACE only
    // (its terms do not depend on auto_threads) and not used.
    plan_threads(view(), sums(), ctrl.num_threads, thread_costs);
  }
  const auto partition = [&](int nthreads, bool budget) {
    if (!ws.tree_par) ws.tree_par = std::make_shared<TreeParallelBuffers>();
    auto &pp = ws.tree_par->partition;
    if (pp.analysis != keep.serial || pp.ns != ns || pp.threads != nthreads ||
        pp.budget != budget) {
      pp.analysis = 0;  // until rebuilt
      pp.ranges = budget
                      ? partition_subtrees(view(), sums(), nthreads, kTreeParallelMinWork, pp.top)
                      : partition_tasks(view(), sums(), nthreads, kTreeParallelMinWork, pp.top);
      pp.analysis = keep.serial;
      pp.ns = ns;
      pp.threads = nthreads;
      pp.budget = budget;
    }
    return &pp;
  };
  const TreeParallelBuffers::Partition *part = nullptr;
  // Below kTreeParallelMinWork both partitions are empty: none.
  if (delay_budget >= 0 && ns >= 2) {
    if (!below_min_work) part = partition(kBudgetNominalThreads, true);
  } else if (want_threads > 1 && !below_min_work) {
    part = partition(want_threads, false);
  }
  const std::vector<std::pair<std::size_t, std::size_t>> no_subtrees;
  const std::vector<std::uint8_t> no_top;
  const auto &subtrees = part != nullptr ? part->ranges : no_subtrees;
  const auto &top = part != nullptr ? part->top : no_top;
  if (want_threads > 1 && subtrees.size() >= 2)
    threads = std::min<int>(want_threads, static_cast<int>(subtrees.size()));

  // Delay budget regions (Control::delay_budget). Without a partition (a
  // tree too light to split): one running count over the whole postorder.
  // Otherwise each subtree counts from zero over its own postorder, and a
  // top front starts from what its children ended with (counts summed,
  // spent if any child was): node_budget holds that end state of every
  // subtree root and top front. All of it depends on the tree alone.
  const bool budget_regions = delay_budget >= 0 && !subtrees.empty();
  std::vector<BudgetState> node_budget;
  if (budget_regions) node_budget.assign(ns, BudgetState{});
  const auto top_in_state = [&](Int s) {
    BudgetState in;
    for (const Int c : keep.children(s)) {
      const BudgetState &cb = node_budget[static_cast<std::size_t>(c)];
      in.count += cb.count;
      in.spent = in.spent || cb.spent;
    }
    return in;
  };

  // The conditional early stop (EarlyWorkStop) under an armed work limit:
  // shared by the threads on the subtree engine.
  std::optional<EarlyGate> early_gate;
  if (work_armed && early.on()) {
    const double at = std::floor(early.ratio * early.free_work);
    std::atomic<long long> *works = nullptr;
    if (threads > 1) {
      TreeParallelBuffers &b = buffers();
      if (b.early_works_cap < ns) {
        b.early_works = std::make_unique<std::atomic<long long>[]>(ns);
        b.early_works_cap = ns;
      }
      works = b.early_works.get();
    }
    early_gate.emplace(
        keep, early,
        at >= 9.0e18 ? std::numeric_limits<long long>::max() : static_cast<long long>(at), works);
  }

  if (threads <= 1) {
    if (det_tc != nullptr) {
      if (!ws.tree_par) ws.tree_par = std::make_shared<TreeParallelBuffers>();
      main.tiles = &ws.tree_par->serial_tiles;
    }
    const FactorContext cx{keep, acp,         ari,         av,          prm,      posdef,
                           fkeep, ws,         passes.data(), front_pass.data(), &node_blocks, nullptr,
                           fallback_static, aptp,             aptp_min_elig, delay_budget,
                           budget_static, det_tc != nullptr ? &tile_aprm : nullptr,
                           ctrl.small_front_kernels, work_cap,
                           nullptr, nullptr, nullptr, nullptr, early_gate ? &*early_gate : nullptr};
    // The budget regions in postorder: subtree ranges are contiguous.
    std::vector<std::size_t> region_end;  // per postorder index: its subtree's last, or npos
    constexpr std::size_t npos = static_cast<std::size_t>(-1);
    if (budget_regions) {
      region_end.assign(keep.postorder.size(), npos);
      for (const auto &[lo, hi] : subtrees)
        for (std::size_t k = lo; k <= hi; ++k) region_end[k] = hi;
    }
    BudgetState budget;  // the running state of the current region
    std::size_t k = 0;
    for (; k < keep.postorder.size(); ++k) {
      const Int sk = keep.postorder[k];
      BudgetState top_state;
      BudgetState *bs = &budget;
      if (budget_regions) {
        if (region_end[k] == npos) {
          top_state = top_in_state(sk);
          bs = &top_state;
        } else if (k == 0 || region_end[k - 1] != region_end[k]) {
          budget = BudgetState{};  // first front of a subtree
        }
      }
      const Int next_s = k + 1 < keep.postorder.size() ? keep.postorder[k + 1] : Int{-1};
      status = process_front(cx, main, 0, sk, true, bs, det_tc, next_s);
      if (budget_regions && (region_end[k] == npos || region_end[k] == k))
        node_budget[static_cast<std::size_t>(sk)] = *bs;
      if (status == ErrorCode::Success && armed && main.st.inertia.negative > neg_limit)
        status = ErrorCode::InertiaExceeded;
      if (status == ErrorCode::Success && work_armed && main.st.front_work > work_cap)
        status = ErrorCode::WorkLimitExceeded;
      if (status != ErrorCode::Success) break;
    }
    if (status == ErrorCode::InertiaExceeded)
      for (std::size_t r = k + 1; r < keep.postorder.size(); ++r)
        st.aborted_flops_skipped +=
            predicted_front_flops(keep, static_cast<std::size_t>(keep.postorder[r]));
  } else {
    if (!ws.tree_par) ws.tree_par = std::make_shared<TreeParallelBuffers>();
    auto &tb = ws.tree_par->threads;
    if (tb.size() < static_cast<std::size_t>(threads)) tb.resize(static_cast<std::size_t>(threads));
    std::vector<FrontPass> tpass(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      auto &b = tb[static_cast<std::size_t>(t)];
      if (b.pos.size() != n) b.pos.assign(n, -1);
      b.steps.clear();
      b.diag.clear();
      FrontPass &P = tpass[static_cast<std::size_t>(t)];
      P.fv = &b.fv;
      P.sri = &b.sri;
      P.steps = &b.steps;
      P.diag = &b.diag;
      P.cbv = &b.cbv;
      P.cbr = &b.cbr;
      P.rows = &b.rows;
      P.cmap = &b.cmap;
      P.tags = &b.tags;
      P.work = &b.work;
      P.pos = b.pos.data();
      P.tiles = &b.tiles;
      passes.push_back(&P);
    }
    node_blocks.resize(ns);
    std::atomic<long long> work_done{0};
    const FactorContext cx{keep, acp,         ari,         av,          prm,      posdef,
                           fkeep, ws,         passes.data(), front_pass.data(), &node_blocks, &ws.tree_par->pool,
                           fallback_static, aptp,             aptp_min_elig, delay_budget,
                           budget_static, tiles_on ? &tile_aprm : nullptr,
                           ctrl.small_front_kernels, work_cap, &work_done,
                           top.empty() ? nullptr : top.data(), &ws.tree_par->root_values,
                           &ws.tree_par->root_rows, early_gate ? &*early_gate : nullptr};
    // `failed` doubles as the stop flag: every thread polls it before each
    // front. With the limit armed, the threads add each front's negative
    // pivots to one shared count; the add that takes it past the limit
    // raises InertiaExceeded (unless another error came first).
    std::atomic<int> failed{static_cast<int>(ErrorCode::Success)};
    std::atomic<int> negatives{0};
    const auto count_negatives = [&](int added) {
      if (added <= 0 || negatives.fetch_add(added, std::memory_order_relaxed) + added <= neg_limit)
        return;
      int ok = static_cast<int>(ErrorCode::Success);
      failed.compare_exchange_strong(ok, static_cast<int>(ErrorCode::InertiaExceeded),
                                     std::memory_order_relaxed);
    };
    // work_limit: the same, with the fronts' work.
    const auto count_work = [&](long long added) {
      if (added <= 0 || work_done.fetch_add(added, std::memory_order_relaxed) + added <= work_cap)
        return;
      int ok = static_cast<int>(ErrorCode::Success);
      failed.compare_exchange_strong(ok, static_cast<int>(ErrorCode::WorkLimitExceeded),
                                     std::memory_order_relaxed);
    };
    // Workers count no BLAS threads outside their own guards: the caller,
    // worker 0, stops counting its outer guard before the others start.
    // The threads persist in ws.pool between phases and factorizations, and
    // count no BLAS threads outside the guards the jobs set.
    const auto run_pool = [threads, &ws](const auto &fn) {
      BlasIdleScope idle;
      const std::function<void(int)> job(std::cref(fn));
      ws.ensure_pool(threads).run(job, /*workers_count_blas=*/false);
    };

    // Phase 1: subtrees (budget regions, or partition_tasks' tasks of
    // merged subtrees), heaviest first, one thread each on serial BLAS.
    // Each thread's step ranges per range become the solve's parallel
    // layout.
    st.threads_used = threads;
    using PhaseClock = std::chrono::steady_clock;
    const auto seconds_since = [](PhaseClock::time_point t) {
      return std::chrono::duration<double>(PhaseClock::now() - t).count();
    };
    ws.tree_par->root_values.reset();
    ws.tree_par->root_rows.reset();
    // The arenas' chunk allocations count as capacity growths
    // (Info::arena_growths), as the pass stacks' growths did when the root
    // blocks lived there.
    const std::size_t arena_growths0 =
        ws.tree_par->root_values.growths() + ws.tree_par->root_rows.growths();
    const auto phase1_start = PhaseClock::now();
    // Thread t takes task t first, then the next one left: the heaviest
    // tasks (the first, heaviest first) land on the same passes in every
    // factorization of an analysis, whose buffers then hold them without
    // growing (which thread wakes first is a race on the pool). A dealt task
    // is claimed, not pinned: a thread that finds no task left takes any
    // dealt task whose thread has not claimed it yet, so a thread that wakes
    // late (by a scheduler quantum on a loaded machine) holds up no task.
    // Only then may a pass take a dealt task not its own, and grow.
    const std::size_t dealt = std::min(subtrees.size(), static_cast<std::size_t>(threads));
    std::vector<std::atomic<bool>> claimed(dealt);
    const auto claim = [&claimed](std::size_t i) {
      return !claimed[i].load(std::memory_order_relaxed) &&
             !claimed[i].exchange(true, std::memory_order_relaxed);
    };
    std::atomic<std::size_t> next{dealt};
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> subtree_steps(
        static_cast<std::size_t>(threads));
    run_pool([&](int tid) {
      BlasSerialGuard serial_blas;
      FrontPass &P = tpass[static_cast<std::size_t>(tid)];
      const std::size_t own = static_cast<std::size_t>(tid);
      const auto running = [&failed] {
        return failed.load(std::memory_order_relaxed) == static_cast<int>(ErrorCode::Success);
      };
      if (own < dealt && tid == g_late_phase1_thread.load(std::memory_order_relaxed)) {
        // Tests: start once another thread has taken this thread's task.
        const auto t0 = std::chrono::steady_clock::now();
        while (!claimed[own].load(std::memory_order_relaxed) &&
               std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
          std::this_thread::yield();
      }
      const auto run_task = [&](std::size_t i) {
        const auto [lo, hi] = subtrees[i];
        const std::size_t first_step = P.steps->size();
        // Delay budget: each subtree from zero over its own postorder, so the
        // outcome does not depend on which thread takes it or when.
        BudgetState budget;
        bool complete = true;
        for (std::size_t k = lo; k <= hi; ++k) {
          if (failed.load(std::memory_order_relaxed) != static_cast<int>(ErrorCode::Success))
            break;
          const int neg0 = P.st.inertia.negative;
          const long long work0 = P.st.front_work;
          const Int next_s = k < hi ? keep.postorder[k + 1] : Int{-1};
          const ErrorCode e = process_front(cx, P, 1 + tid, keep.postorder[k], true,
                                            budget_regions ? &budget : nullptr, det_tc, next_s);
          if (e != ErrorCode::Success) {
            failed.store(static_cast<int>(e), std::memory_order_relaxed);
            complete = false;
            break;
          }
          P.diag_marks.emplace_back(keep.postorder[k], P.diag->size());
          if (armed) count_negatives(P.st.inertia.negative - neg0);
          if (work_armed) count_work(P.st.front_work - work0);
        }
        // The subtree root's end state, read by the top part after the join.
        if (budget_regions && complete)
          node_budget[static_cast<std::size_t>(keep.postorder[hi])] = budget;
        subtree_steps[static_cast<std::size_t>(tid)].emplace_back(first_step, P.steps->size());
      };
      if (own < dealt && running() && claim(own)) run_task(own);
      while (running()) {
        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= subtrees.size()) break;
        run_task(i);
      }
      for (std::size_t i = 0; i < dealt && running(); ++i)
        if (claim(i)) {
          g_dealt_tasks_taken_over.fetch_add(1, std::memory_order_relaxed);
          run_task(i);
        }
    });
    st.subtree_seconds = seconds_since(phase1_start);
    main.growths += static_cast<int64_t>(ws.tree_par->root_values.growths() +
                                         ws.tree_par->root_rows.growths() - arena_growths0);
    const auto phase2_start = PhaseClock::now();
    std::vector<std::size_t> phase1_steps(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t)
      phase1_steps[static_cast<std::size_t>(t)] = tpass[static_cast<std::size_t>(t)].steps->size();

    // Phase 2: the top of the tree as a task DAG (TopScheduler). A node is
    // ready once its children are factored; ready nodes run concurrently
    // under a budget of `threads` BLAS threads in total (oversubscribing
    // MKL's spinning teams is ruinous): a front takes its share of the free
    // threads among the fronts ready with it, so sibling separators run side
    // by side and the root, alone, gets them all. The thread that factors a
    // front's last child continues with that front itself: the top part is
    // mostly long chains of small fronts, which then run without hand-offs
    // or wake-ups between threads.
    // Stacks are no longer popped (a worker's blocks are consumed out of
    // order).
    std::vector<Int> top_nodes;            // local index -> supernode
    std::vector<Int> top_local(ns, -1);    // supernode -> local index (-1: not top)
    for (std::size_t si = 0; si < ns; ++si) {
      if (!top[si]) continue;
      top_local[si] = static_cast<Int>(top_nodes.size());
      top_nodes.push_back(static_cast<Int>(si));
    }
    std::vector<Int> top_parent(top_nodes.size(), -1);
    for (std::size_t i = 0; i < top_nodes.size(); ++i) {
      const Int par = keep.supernodes[static_cast<std::size_t>(top_nodes[i])].parent;
      if (par >= 0) top_parent[i] = top_local[static_cast<std::size_t>(par)];
    }
    // 2D tile tasks (Control::top_tiles): a front given blas_threads threads
    // offers blas_threads - 1 places to the scheduler, taken by the workers
    // that the BLAS-thread budget leaves idle; they run its tasks on serial
    // BLAS.
    TopScheduler sched(std::move(top_parent), threads);
    // The passes as lanes (PassLanes): a chain of top fronts runs on a pass
    // that already has the room its next front assembles in (factor values
    // past the pass's fill), whichever thread runs it, so the root fronts
    // stay in the buffers they grew in an earlier factorization. A thread
    // starts on its own pass (its phase-1 blocks are there) when that has
    // the room. lane_of: per thread, the pass its current chain holds.
    PassLanes lanes(threads);
    std::vector<int> lane_of(static_cast<std::size_t>(threads), -1);
    const auto lane_room = [&tpass](int l) -> std::size_t {
      const FrontPass &Q = tpass[static_cast<std::size_t>(l)];
      return Q.fv->size() > Q.fo ? Q.fv->size() - Q.fo : 0;
    };
    // Each top front's pass and step index in it (by local index).
    std::vector<int> top_pass(top_nodes.size(), 0);
    std::vector<std::size_t> top_step(top_nodes.size(), 0);
    if (failed.load() == static_cast<int>(ErrorCode::Success) && !top_nodes.empty()) {
      run_pool([&](int tid) {
        // false stops every thread: an error here, or a stop raised by
        // another thread (an error, or a limit crossed by the counts).
        int &lane = lane_of[static_cast<std::size_t>(tid)];
        const auto run = [&](Int local, int blas_threads, int) {
          if (failed.load() != static_cast<int>(ErrorCode::Success)) return false;
          const Int s = top_nodes[static_cast<std::size_t>(local)];
          // The front's rows at most (its children's delays are final), and
          // the factor values it is assembled in.
          std::size_t f = static_cast<std::size_t>(keep.front_size(s));
          for (const Int c : keep.children(s))
            f += static_cast<std::size_t>(ws.cb_delayed[static_cast<std::size_t>(c)]);
          lane = lanes.pick(lane, tid, f * f, lane_room);
          FrontPass &P = tpass[static_cast<std::size_t>(lane)];
          // Delay budget of a top front: its children's end states, all
          // final once it is ready (written before the scheduler recorded
          // their completion, under its lock, or by this thread).
          BudgetState top_state;
          if (budget_regions) top_state = top_in_state(s);
          const int neg0 = P.st.inertia.negative;
          const long long work0 = P.st.front_work;
          ErrorCode e;
          {
            // Deterministic mode: BLAS on one thread whatever the front's
            // share (its tile tasks still recruit blas_threads - 1 helpers).
            BlasThreadGuard blas(det ? 1 : blas_threads);
            const TileCtx tc{blas_threads, &sched};
            e = process_front(cx, P, 1 + lane, s, false, budget_regions ? &top_state : nullptr, &tc);
          }
          if (e != ErrorCode::Success) {
            failed.store(static_cast<int>(e));
            return false;
          }
          if (armed) count_negatives(P.st.inertia.negative - neg0);
          if (work_armed) count_work(P.st.front_work - work0);
          top_pass[static_cast<std::size_t>(local)] = lane;
          top_step[static_cast<std::size_t>(local)] = P.steps->size() - 1;
          ++P.st.top_part_fronts;
          P.diag_marks.emplace_back(s, P.diag->size());
          if (budget_regions) node_budget[static_cast<std::size_t>(s)] = top_state;
          // The counts above may have raised the stop.
          return failed.load() == static_cast<int>(ErrorCode::Success);
        };
        sched.work(tid, run, [&](int) {
          lanes.release(lane);
          lane = -1;
        });
      });
    }
    // Top fronts in completion order, which is topological: the solve
    // layout of the top part.
    struct Done {
      int pass = 0;
      std::size_t step = 0;
      Int node = -1;
    };
    std::vector<Done> done_order;
    done_order.reserve(sched.completed().size());
    for (const auto &c : sched.completed()) {
      const std::size_t local = static_cast<std::size_t>(c.first);
      done_order.push_back({top_pass[local], top_step[local], top_nodes[local]});
    }
    st.top_part_queue_pops = sched.queue_pops();
    st.top_seconds = seconds_since(phase2_start);
    status = static_cast<ErrorCode>(failed.load());
    st.parallel_subtrees = static_cast<int>(subtrees.size());

    if (status != ErrorCode::Success) {
      // Partial counts (the threads are joined). A front never started
      // still has front_pass 0: the threads' passes are numbered from 1.
      for (int t = 0; t < threads; ++t) add_stats(st, tpass[static_cast<std::size_t>(t)].st);
      if (status == ErrorCode::InertiaExceeded)
        for (std::size_t si = 0; si < ns; ++si)
          if (front_pass[si] == 0) st.aborted_flops_skipped += predicted_front_flops(keep, si);
    } else {
      // Phase 3: concatenate the per-thread factors (copied in parallel) and
      // relocate offsets. Solve order: every subtree (each complete and
      // independent), then the top in completion order, which is
      // topological since a front starts only after its children finish.
      std::vector<std::size_t> fbase(static_cast<std::size_t>(threads) + 1, 0);
      std::vector<std::size_t> rbase(static_cast<std::size_t>(threads) + 1, 0);
      for (int t = 0; t < threads; ++t) {
        const std::size_t ti = static_cast<std::size_t>(t);
        fbase[ti + 1] = fbase[ti] + tpass[ti].fo;
        rbase[ti + 1] = rbase[ti] + tpass[ti].ro;
      }
      main.fo = fbase[static_cast<std::size_t>(threads)];
      main.ro = rbase[static_cast<std::size_t>(threads)];
      const auto phase3_start = PhaseClock::now();
      grow(fv, main.fo, main.growths);
      grow(sri, main.ro, main.growths);
      const bool in_pieces =
          main.fo * sizeof(double) + main.ro * sizeof(Int) <= kCachedGatherBytes;
      run_pool([&](int tid) {
        const std::size_t ti = static_cast<std::size_t>(tid);
        const FrontPass &P = tpass[ti];
        assert(P.fo <= P.fv->size() && fbase[ti] + P.fo <= fv.size());
        assert(P.ro <= P.sri->size() && rbase[ti] + P.ro <= sri.size());
        gather_copy(P.fv->data(), P.fo, fv.data() + fbase[ti], in_pieces);
        gather_copy(P.sri->data(), P.ro, sri.data() + rbase[ti], in_pieces);
      });
      st.gather_seconds = seconds_since(phase3_start);
      const auto relocated = [&](int t, SolveStep step) {
        step.factor_offset += static_cast<LongInt>(fbase[static_cast<std::size_t>(t)]);
        step.row_offset += static_cast<LongInt>(rbase[static_cast<std::size_t>(t)]);
        return step;
      };
      std::vector<std::size_t> post_pos(ns, 0);
      for (std::size_t k = 0; k < keep.postorder.size(); ++k)
        post_pos[static_cast<std::size_t>(keep.postorder[k])] = k;
      for (int t = 0; t < threads; ++t) {
        const std::size_t ti = static_cast<std::size_t>(t);
        const auto &steps = *tpass[ti].steps;
        const auto base = static_cast<LongInt>(fkeep.solve_steps.size());
        if (!det) {
          for (const auto &[b, e] : subtree_steps[ti])
            fkeep.solve_subtrees.emplace_back(base + static_cast<LongInt>(b),
                                              base + static_cast<LongInt>(e));
          for (std::size_t k = 0; k < phase1_steps[ti]; ++k)
            fkeep.solve_steps.push_back(relocated(t, steps[k]));
        }
        add_stats(st, tpass[ti].st);
        main.growths += tpass[ti].growths;
      }
      // D entries in elimination order: fronts by their position in the
      // postorder (children before parents), each front's entries in pivot
      // order. Thread assignment and the top phase's completion order vary
      // run to run; this order does not, and only the solve of a non-inline
      // factor reads it (entry by entry, in any order).
      {
        // Each front's entries [b, e) in its pass, by its position in the
        // postorder (b = e: none).
        struct Chunk {
          int pass = 0;
          std::size_t b = 0, e = 0;
        };
        std::vector<Chunk> chunks(keep.postorder.size());
        for (int t = 0; t < threads; ++t) {
          std::size_t b = 0;
          for (const auto &[node, end] : tpass[static_cast<std::size_t>(t)].diag_marks) {
            if (end > b) chunks[post_pos[static_cast<std::size_t>(node)]] = {t, b, end};
            b = end;
          }
        }
        for (const Chunk &c : chunks) {
          if (c.e == c.b) continue;
          const auto &d = *tpass[static_cast<std::size_t>(c.pass)].diag;
          fkeep.diag_entries.insert(fkeep.diag_entries.end(), d.begin() + static_cast<std::ptrdiff_t>(c.b),
                                    d.begin() + static_cast<std::ptrdiff_t>(c.e));
        }
      }
      if (det) {
        // Deterministic mode: every front's solve step in postorder, the
        // order of the serial factorization, and no parallel solve layout
        // (the solve runs serially): the solve then does the same
        // operations in the same order whatever the thread count. A pass's
        // k-th step is the front of its k-th diag mark.
        struct StepRef {
          std::size_t post;
          int pass;
          std::size_t k;
        };
        std::vector<StepRef> refs;
        refs.reserve(ns);
        for (int t = 0; t < threads; ++t) {
          const auto &marks = tpass[static_cast<std::size_t>(t)].diag_marks;
          assert(marks.size() == tpass[static_cast<std::size_t>(t)].steps->size());
          for (std::size_t k = 0; k < marks.size(); ++k)
            refs.push_back({post_pos[static_cast<std::size_t>(marks[k].first)], t, k});
        }
        std::sort(refs.begin(), refs.end(),
                  [](const StepRef &a, const StepRef &b) { return a.post < b.post; });
        for (const StepRef &r : refs)
          fkeep.solve_steps.push_back(
              relocated(r.pass, (*tpass[static_cast<std::size_t>(r.pass)].steps)[r.k]));
      } else {
        fkeep.solve_top_begin = static_cast<LongInt>(fkeep.solve_steps.size());
        // The solve takes subtrees in this order: heaviest (stored entries)
        // first, so the last ones to start are small.
        {
          std::vector<std::pair<double, std::pair<LongInt, LongInt>>> by_work;
          by_work.reserve(fkeep.solve_subtrees.size());
          for (const auto &r : fkeep.solve_subtrees) {
            double w = 0.0;
            for (LongInt k = r.first; k < r.second; ++k) {
              const SolveStep &step = fkeep.solve_steps[static_cast<std::size_t>(k)];
              w += static_cast<double>(step.p) * step.f;
            }
            by_work.push_back({w, r});
          }
          std::sort(by_work.begin(), by_work.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });
          for (std::size_t i = 0; i < by_work.size(); ++i) fkeep.solve_subtrees[i] = by_work[i].second;
        }
        for (const Done &d : done_order)
          fkeep.solve_steps.push_back(
              relocated(d.pass, (*tpass[static_cast<std::size_t>(d.pass)].steps)[d.step]));
        // Top-part tree over solve steps (parents are top nodes too).
        {
          std::vector<Int> top_pos(ns, -1);
          for (std::size_t i = 0; i < done_order.size(); ++i)
            top_pos[static_cast<std::size_t>(done_order[i].node)] = static_cast<Int>(i);
          fkeep.solve_top_parent.resize(done_order.size());
          for (std::size_t i = 0; i < done_order.size(); ++i) {
            const Int par = keep.supernodes[static_cast<std::size_t>(done_order[i].node)].parent;
            fkeep.solve_top_parent[i] = par >= 0 ? top_pos[static_cast<std::size_t>(par)] : -1;
          }
        }
      }
      for (std::size_t si = 0; si < ns; ++si)
        fkeep.factor_col_ptr[si] +=
            static_cast<LongInt>(fbase[static_cast<std::size_t>(front_pass[si] - 1)]);
    }
  }

  ws.capacity_growths_last_call += main.growths;
  ws.front_heap_allocations_last_call += main.growths;
  add_stats(st, main.st);

  // Leave the stack bookkeeping consistent for the next planning pass.
  if (status != ErrorCode::Success) {
    fkeep.solve_subtrees.clear();
    fkeep.solve_threads = 1;
    std::fill(ws.cb_size.begin(), ws.cb_size.end(), Int{0});
    std::fill(ws.cb_delayed.begin(), ws.cb_delayed.end(), Int{0});
    return status;
  }
  fkeep.factor_col_ptr[ns] = static_cast<LongInt>(main.fo);
  fv.resize(main.fo);
  sri.resize(main.ro);
  if (!fkeep.solve_subtrees.empty()) {
    // Pivot rows of the top part (delayed subtree pivots included): the
    // parallel forward solve sends subtree updates to them through
    // per-thread buffers.
    if (fkeep.solve_top_slot.size() != n) fkeep.solve_top_slot.assign(n, -1);
    else std::fill(fkeep.solve_top_slot.begin(), fkeep.solve_top_slot.end(), Int{-1});
    std::vector<double> top_work;  // p f per step, the solve's cost model
    top_work.reserve(fkeep.solve_steps.size() - static_cast<std::size_t>(fkeep.solve_top_begin));
    for (std::size_t k = static_cast<std::size_t>(fkeep.solve_top_begin);
         k < fkeep.solve_steps.size(); ++k) {
      const SolveStep &step = fkeep.solve_steps[k];
      const Int *rows = sri.data() + step.row_offset;
      for (int i = 0; i < step.p; ++i) {
        fkeep.solve_top_slot[static_cast<std::size_t>(rows[i])] =
            static_cast<Int>(fkeep.solve_top_rows.size());
        fkeep.solve_top_rows.push_back(rows[i]);
      }
      top_work.push_back(static_cast<double>(step.p) * step.f);
    }
    // The solve's own thread count: its subtrees and top part weigh
    // differently from the factorization's (p f, not f^2 p). auto_threads
    // off: the factorization's thread count (the plan is evaluated for
    // SMF_BENCH_PLAN_TRACE only).
    if (ctrl.auto_threads || plan_trace_enabled()) {
      std::vector<double> sub_work;
      sub_work.reserve(fkeep.solve_subtrees.size());
      for (const auto &[b, e] : fkeep.solve_subtrees) {
        double w = 0.0;
        for (auto k = static_cast<std::size_t>(b); k < static_cast<std::size_t>(e); ++k)
          w += static_cast<double>(fkeep.solve_steps[k].p) * fkeep.solve_steps[k].f;
        sub_work.push_back(w);
      }
      fkeep.solve_threads = plan_solve_threads(
          sub_work, top_work, fkeep.solve_top_parent, fkeep.solve_top_rows.size(),
          ctrl.num_threads,
          SolveCosts{ctrl.solve_cost_per_thread, ctrl.solve_cost_per_top_step,
                     ctrl.solve_cost_per_top_row});
    }
    if (!ctrl.auto_threads) fkeep.solve_threads = threads;
    // A solve runs at most one thread per subtree (it takes min(want,
    // subtrees)): store what a solve can use.
    fkeep.solve_threads =
        std::min(fkeep.solve_threads, static_cast<int>(fkeep.solve_subtrees.size()));
  }
  return ErrorCode::Success;
}

} // namespace smf
