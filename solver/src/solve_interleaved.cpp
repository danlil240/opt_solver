#include "smf/solve_interleaved.hpp"

#include "smf/blas_thread_guard.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/solve_diag.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace smf {

namespace {

// V vectors of four doubles: one interleaved row of up to 4 V right-hand
// sides (unused lanes hold zeros and stay zero).
template <int V> struct Lanes {
  static constexpr int kWidth = 4 * V;
#if defined(__AVX2__)
  __m256d v[V];
  static Lanes load(const double *p) {
    Lanes a;
    for (int k = 0; k < V; ++k) a.v[k] = _mm256_loadu_pd(p + 4 * k);
    return a;
  }
  void store(double *p) const {
    for (int k = 0; k < V; ++k) _mm256_storeu_pd(p + 4 * k, v[k]);
  }
  static Lanes zero() {
    Lanes a;
    for (int k = 0; k < V; ++k) a.v[k] = _mm256_setzero_pd();
    return a;
  }
  Lanes plus(const Lanes &y) const {
    Lanes r;
    for (int k = 0; k < V; ++k) r.v[k] = _mm256_add_pd(v[k], y.v[k]);
    return r;
  }
  // this - l * x
  Lanes minus(double l, const Lanes &x) const {
    const __m256d b = _mm256_set1_pd(l);
    Lanes r;
    for (int k = 0; k < V; ++k) r.v[k] = _mm256_fnmadd_pd(b, x.v[k], v[k]);
    return r;
  }
  Lanes times(double s) const {
    const __m256d b = _mm256_set1_pd(s);
    Lanes r;
    for (int k = 0; k < V; ++k) r.v[k] = _mm256_mul_pd(b, v[k]);
    return r;
  }
  Lanes over(double s) const {
    const __m256d b = _mm256_set1_pd(s);
    Lanes r;
    for (int k = 0; k < V; ++k) r.v[k] = _mm256_div_pd(v[k], b);
    return r;
  }
  // a * this + b * y
  Lanes combine(double a, double b, const Lanes &y) const {
    const __m256d va = _mm256_set1_pd(a), vb = _mm256_set1_pd(b);
    Lanes r;
    for (int k = 0; k < V; ++k) r.v[k] = _mm256_fmadd_pd(vb, y.v[k], _mm256_mul_pd(va, v[k]));
    return r;
  }
#else
  double v[4 * V];
  static Lanes load(const double *p) {
    Lanes a;
    for (int k = 0; k < 4 * V; ++k) a.v[k] = p[k];
    return a;
  }
  void store(double *p) const {
    for (int k = 0; k < 4 * V; ++k) p[k] = v[k];
  }
  static Lanes zero() {
    Lanes a;
    for (int k = 0; k < 4 * V; ++k) a.v[k] = 0.0;
    return a;
  }
  Lanes plus(const Lanes &y) const {
    Lanes r;
    for (int k = 0; k < 4 * V; ++k) r.v[k] = v[k] + y.v[k];
    return r;
  }
  Lanes minus(double l, const Lanes &x) const {
    Lanes r;
    for (int k = 0; k < 4 * V; ++k) r.v[k] = v[k] - l * x.v[k];
    return r;
  }
  Lanes times(double s) const {
    Lanes r;
    for (int k = 0; k < 4 * V; ++k) r.v[k] = s * v[k];
    return r;
  }
  Lanes over(double s) const {
    Lanes r;
    for (int k = 0; k < 4 * V; ++k) r.v[k] = v[k] / s;
    return r;
  }
  Lanes combine(double a, double b, const Lanes &y) const {
    Lanes r;
    for (int k = 0; k < 4 * V; ++k) r.v[k] = a * v[k] + b * y.v[k];
    return r;
  }
#endif
};

// One right-hand side: the kernels below on x itself, in place.
struct Lane1 {
  static constexpr int kWidth = 1;
  double v;
  static Lane1 load(const double *p) { return {*p}; }
  void store(double *p) const { *p = v; }
  static Lane1 zero() { return {0.0}; }
  Lane1 plus(const Lane1 &y) const { return {v + y.v}; }
  Lane1 minus(double l, const Lane1 &x) const { return {v - l * x.v}; }
  Lane1 times(double s) const { return {s * v}; }
  Lane1 over(double s) const { return {v / s}; }
  Lane1 combine(double a, double b, const Lane1 &y) const { return {a * v + b * y.v}; }
};

// One small front of L^-1 on the interleaved rows X (row stride S = 4 V):
// column by column, every entry of L applied to all lanes at once.
template <class Ln>
void forward_small(const double *L, const Int *rows, int p, int f, bool posdef, double *X) {
  constexpr int S = Ln::kWidth;
  for (int j = 0; j < p; ++j) {
    const double *col = L + static_cast<std::ptrdiff_t>(j) * f;
    double *xj = X + static_cast<std::ptrdiff_t>(rows[j]) * S;
    Ln a = Ln::load(xj);
    if (posdef) {
      a = a.over(col[j]);
      a.store(xj);
    }
    for (int i = j + 1; i < f; ++i) {
      double *xi = X + static_cast<std::ptrdiff_t>(rows[i]) * S;
      Ln::load(xi).minus(col[i], a).store(xi);
    }
  }
}

// One small front of D^-1 L^-T: D^-1 on the pivot rows (indefinite), then
// pivot j = p-1 .. 0 takes x_j -= L(i, j) x_i over all later rows i (the
// later pivots of this front are final by then, the extended rows belong to
// ancestors).
template <class Ln>
void backward_small(const double *L, const Int *rows, int p, int f, bool posdef,
                    const double *dinv, const std::int8_t *dkind, double *X) {
  constexpr int S = Ln::kWidth;
  if (!posdef) {
    for (int c = 0; c < p;) {
      const double *e = dinv + 3 * c;
      double *x0 = X + static_cast<std::ptrdiff_t>(rows[c]) * S;
      if (dkind[c] == 2) {
        double *x1 = X + static_cast<std::ptrdiff_t>(rows[c + 1]) * S;
        const Ln a = Ln::load(x0), b = Ln::load(x1);
        a.combine(e[0], e[1], b).store(x0);
        a.combine(e[1], e[2], b).store(x1);
        c += 2;
      } else {
        Ln::load(x0).times(e[0]).store(x0);
        ++c;
      }
    }
  }
  for (int j = p - 1; j >= 0; --j) {
    const double *col = L + static_cast<std::ptrdiff_t>(j) * f;
    double *xj = X + static_cast<std::ptrdiff_t>(rows[j]) * S;
    // Two accumulators: the FMA latency, not the loads, bounds one chain.
    Ln acc = Ln::load(xj), acc1 = Ln::zero();
    int i = j + 1;
    for (; i + 1 < f; i += 2) {
      acc = acc.minus(col[i], Ln::load(X + static_cast<std::ptrdiff_t>(rows[i]) * S));
      acc1 = acc1.minus(col[i + 1], Ln::load(X + static_cast<std::ptrdiff_t>(rows[i + 1]) * S));
    }
    if (i < f)
      acc = acc.minus(col[i], Ln::load(X + static_cast<std::ptrdiff_t>(rows[i]) * S));
    acc = acc.plus(acc1);
    if (posdef)
      acc = acc.over(col[j]);
    acc.store(xj);
  }
}

// W (f x nr, column-major) <- rows of X, and back.
void gather_block(const double *X, int S, const Int *rows, int f, int nr, double *W) {
  for (int i = 0; i < f; ++i) {
    const double *xi = X + static_cast<std::ptrdiff_t>(rows[i]) * S;
    for (int r = 0; r < nr; ++r)
      W[static_cast<std::ptrdiff_t>(r) * f + i] = xi[r];
  }
}

void scatter_block(const double *W, int f, int nr, const Int *rows, int count, int S,
                   double *X) {
  for (int i = 0; i < count; ++i) {
    double *xi = X + static_cast<std::ptrdiff_t>(rows[i]) * S;
    for (int r = 0; r < nr; ++r)
      xi[r] = W[static_cast<std::ptrdiff_t>(r) * f + i];
  }
}

// A large front (or one with a cached L11^-1) of L^-1 with BLAS-3, as
// forward_front_block does it.
void forward_large(const double *L, const double *Linv, const Int *rows, int p, int f,
                   bool posdef, int nr, int S, double *X, std::vector<double> &wbuf,
                   std::vector<double> &tbuf) {
  const int q = f - p;
  wbuf.resize(std::max(wbuf.size(), static_cast<std::size_t>(f) * static_cast<std::size_t>(nr)));
  double *W = wbuf.data();
  gather_block(X, S, rows, f, nr, W);
  if (Linv) {
    tbuf.resize(std::max(tbuf.size(), static_cast<std::size_t>(p) * static_cast<std::size_t>(nr)));
    smf_dgemm(tbuf.data(), p, nr, p, Linv, p, W, f, p, 1.0, 0.0);
    for (int r = 0; r < nr; ++r)
      std::copy_n(tbuf.data() + static_cast<std::ptrdiff_t>(r) * p, p,
                  W + static_cast<std::ptrdiff_t>(r) * f);
  } else {
    smf_dtrsm_left_lower(W, p, nr, L, f, f, /*trans=*/false, /*unit=*/!posdef);
  }
  if (q > 0)
    smf_dgemm(W + p, q, nr, p, L + p, f, W, f, f, -1.0, 1.0);
  scatter_block(W, f, nr, rows, f, S, X);
}

// D^-1 for one gathered column's pivot rows (FactorKeep::dinv entries).
void apply_dinv_column(double *w, int p, const double *d, const std::int8_t *kind) {
  for (int c = 0; c < p;) {
    const double *e = d + 3 * c;
    if (kind[c] == 2) {
      const double v0 = w[c], v1 = w[c + 1];
      w[c] = e[0] * v0 + e[1] * v1;
      w[c + 1] = e[1] * v0 + e[2] * v1;
      c += 2;
    } else {
      w[c] *= e[0];
      ++c;
    }
  }
}

// A large front of D^-1 L^-T with BLAS-3, as backward_front_block does it.
void backward_large(const double *L, const double *Linv, const Int *rows, int p, int f,
                    bool posdef, const double *dinv, const std::int8_t *dkind, int nr, int S,
                    double *X, std::vector<double> &wbuf, std::vector<double> &tbuf) {
  const int q = f - p;
  wbuf.resize(std::max(wbuf.size(), static_cast<std::size_t>(f) * static_cast<std::size_t>(nr)));
  double *W = wbuf.data();
  gather_block(X, S, rows, f, nr, W);
  if (!posdef)
    for (int r = 0; r < nr; ++r)
      apply_dinv_column(W + static_cast<std::ptrdiff_t>(r) * f, p, dinv, dkind);
  if (q > 0)
    smf_dgemm_tn(W, p, nr, q, L + p, f, W + p, f, f, -1.0, 1.0);
  if (Linv) {
    tbuf.resize(std::max(tbuf.size(), static_cast<std::size_t>(p) * static_cast<std::size_t>(nr)));
    smf_dgemm_tn(tbuf.data(), p, nr, p, Linv, p, W, f, p, 1.0, 0.0);
    for (int r = 0; r < nr; ++r)
      std::copy_n(tbuf.data() + static_cast<std::ptrdiff_t>(r) * p, p,
                  W + static_cast<std::ptrdiff_t>(r) * f);
  } else {
    smf_dtrsm_left_lower(W, p, nr, L, f, f, /*trans=*/true, /*unit=*/!posdef);
  }
  scatter_block(W, f, nr, rows, p, S, X);
}

template <class Ln>
void solve_chunk(const FactorKeep &fkeep, double *b, int n, int nr, int ldb,
                 SolveWorkspace &ws) {
  constexpr int S = Ln::kWidth;
  std::vector<double> &Xv = ws.ilv;
  if (Xv.size() < static_cast<std::size_t>(n) * S)
    Xv.resize(static_cast<std::size_t>(n) * S);
  double *X = Xv.data();
  const int *perm = fkeep.perm.data();
  const bool has_scale = !fkeep.scale.empty() && static_cast<int>(fkeep.scale.size()) == n;
  const double *s = has_scale ? fkeep.scale.data() : nullptr;
  // X = P^T S b, interleaved; the unused lanes zero.
  for (int i = 0; i < n; ++i) {
    double *xi = X + static_cast<std::ptrdiff_t>(i) * S;
    const int pi = perm[i];
    const double si = s ? s[pi] : 1.0;
    int r = 0;
    for (; r < nr; ++r)
      xi[r] = b[static_cast<std::ptrdiff_t>(r) * ldb + pi] * si;
    for (; r < S; ++r)
      xi[r] = 0.0;
  }

  const bool posdef = fkeep.is_posdef;
  const bool has_inv = fkeep.inv_diag_offset.size() == fkeep.solve_steps.size();
  const std::size_t ns = fkeep.solve_steps.size();
  const auto front = [&](std::size_t k, const double *&L, const double *&Linv, const Int *&rows,
                         int &p, int &f) {
    const SolveStep &st = fkeep.solve_steps[k];
    rows = fkeep.solve_row_indices.data() + st.row_offset;
    L = fkeep.factor_values.data() + st.factor_offset;
    Linv = (has_inv && fkeep.inv_diag_offset[k] >= 0)
               ? fkeep.inv_diag.data() + fkeep.inv_diag_offset[k]
               : nullptr;
    p = st.p;
    f = st.f;
  };

  for (std::size_t k = 0; k < ns; ++k) {
    const double *L, *Linv;
    const Int *rows;
    int p, f;
    front(k, L, Linv, rows, p, f);
    if (p <= 0)
      continue;
    if (p < kInvDiagMinPivots && !Linv)
      forward_small<Ln>(L, rows, p, f, posdef, X);
    else
      forward_large(L, Linv, rows, p, f, posdef, nr, S, X, ws.loc, ws.ext);
  }
  for (std::size_t k = ns; k-- > 0;) {
    const double *L, *Linv;
    const Int *rows;
    int p, f;
    front(k, L, Linv, rows, p, f);
    if (p <= 0)
      continue;
    const double *dinv = posdef ? nullptr : fkeep.dinv.data() + 3 * fkeep.dinv_offset[k];
    const std::int8_t *dkind = posdef ? nullptr : fkeep.dinv_kind.data() + fkeep.dinv_offset[k];
    if (p < kInvDiagMinPivots && !Linv)
      backward_small<Ln>(L, rows, p, f, posdef, dinv, dkind, X);
    else
      backward_large(L, Linv, rows, p, f, posdef, dinv, dkind, nr, S, X, ws.loc, ws.ext);
  }

  // b = S P x.
  for (int r = 0; r < nr; ++r) {
    double *dst = b + static_cast<std::ptrdiff_t>(r) * ldb;
    if (s)
      for (int i = 0; i < n; ++i)
        dst[perm[i]] = X[static_cast<std::ptrdiff_t>(i) * S + r] * s[perm[i]];
    else
      for (int i = 0; i < n; ++i)
        dst[perm[i]] = X[static_cast<std::ptrdiff_t>(i) * S + r];
  }
}

} // namespace

bool interleaved_solve_supported(const FactorKeep &fkeep) {
  if (fkeep.solve_steps.empty() || fkeep.use_banded_spd || fkeep.factor_values.empty())
    return false;
  if (static_cast<int>(fkeep.perm.size()) != fkeep.n)
    return false;
  if (fkeep.is_posdef)
    return true;
  return fkeep.inline_d && fkeep.dinv_offset.size() == fkeep.solve_steps.size();
}

void solve_full_interleaved(const FactorKeep &fkeep, double *b, int n, int nrhs, int ldb,
                            SolveWorkspace &ws) {
  for (int c0 = 0; c0 < nrhs; c0 += kInterleavedMaxRhs) {
    const int nr = std::min(kInterleavedMaxRhs, nrhs - c0);
    double *bc = b + static_cast<std::ptrdiff_t>(c0) * ldb;
    if (nr == 1)
      solve_chunk<Lane1>(fkeep, bc, n, nr, ldb, ws);
    else if (nr <= 4)
      solve_chunk<Lanes<1>>(fkeep, bc, n, nr, ldb, ws);
    else
      solve_chunk<Lanes<2>>(fkeep, bc, n, nr, ldb, ws);
  }
}

} // namespace smf
