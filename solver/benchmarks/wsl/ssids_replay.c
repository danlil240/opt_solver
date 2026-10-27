/*
 * ssids_replay: SPRAL SSIDS (CPU) driver for the smf benchmark harness.
 *
 * Same command line and the same JSONL record as bench_replay (cold and
 * warm_sequence modes, refinement on the ipopt track), so tools/bench/run.py
 * can schedule it like any other solver. It runs inside WSL2 Ubuntu because
 * SPRAL does not build natively on Windows; tools/bench/run_wsl.py launches it
 * through wsl.exe and translates the Windows paths. Build recipe:
 * tools/bench/wsl_ssids.md.
 *
 * Inputs: --mtx FILE (Matrix Market, or the .smfbin cache bench_replay writes
 * next to it, which is byte-compatible) and optionally --seq-values FILE (raw
 * doubles, nnz per recorded factorization). Built-in cases are not generated
 * here: run_wsl.py exports them with `bench_replay --export-mtx`.
 *
 * SSIDS configuration (the same knobs as the other adapters):
 *   u            = --pivot-u (default 0.01; 1e-8 on the ipopt track)
 *   SPD matrices : Cholesky (posdef = true), METIS ordering, no scaling
 *   indefinite   : LDL^T, matching-based ordering with its scaling
 *                  (ordering = 2, scaling = 3), SPRAL's recommended setting for
 *                  hard indefinite problems; --ssids-ordering / --ssids-scaling
 *                  override.
 * SSIDS needs OMP_CANCELLATION=TRUE and OMP_PROC_BIND=TRUE (run_wsl.py sets
 * them). Threads: OMP_NUM_THREADS (= --threads).
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "spral_ssids.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------- matrix */

typedef struct {
    int n;
    int64_t nnz;
    int64_t *ptr; /* n + 1, 0-based */
    int *row;     /* nnz, lower triangle, sorted within columns */
    double *val;  /* nnz */
} Csc;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
}

static int64_t peak_rss_bytes(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    int64_t kb = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            kb = atoll(line + 6);
            break;
        }
    }
    fclose(f);
    return kb < 0 ? -1 : kb * 1024;
}

static const char kBinMagic[8] = {'S', 'M', 'F', 'B', 'I', 'N', '0', '1'};

static bool newer_or_equal(const char *a, const char *b) {
    struct stat sa, sb;
    if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return false;
    return sa.st_mtime >= sb.st_mtime;
}

/* bench_replay's <mtx>.smfbin: magic[8], int64 n, int64 nnz, int32 col_ptr[n+1],
 * int32 row_idx[nnz], double values[nnz]. */
static bool load_bin_cache(const char *mtx, Csc *A) {
    char path[4096];
    FILE *f;
    char magic[8];
    int64_t n, nnz;
    int32_t *cp = NULL;
    int64_t i;
    snprintf(path, sizeof path, "%s.smfbin", mtx);
    if (!newer_or_equal(path, mtx)) return false;
    f = fopen(path, "rb");
    if (!f) return false;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, kBinMagic, 8) != 0 || fread(&n, 8, 1, f) != 1 ||
        fread(&nnz, 8, 1, f) != 1 || n < 0 || nnz < 0) {
        fclose(f);
        return false;
    }
    cp = malloc((size_t)(n + 1) * sizeof(int32_t));
    A->n = (int)n;
    A->nnz = nnz;
    A->ptr = malloc((size_t)(n + 1) * sizeof(int64_t));
    A->row = malloc((size_t)(nnz ? nnz : 1) * sizeof(int));
    A->val = malloc((size_t)(nnz ? nnz : 1) * sizeof(double));
    if (fread(cp, sizeof(int32_t), (size_t)(n + 1), f) != (size_t)(n + 1) ||
        fread(A->row, sizeof(int32_t), (size_t)nnz, f) != (size_t)nnz ||
        fread(A->val, sizeof(double), (size_t)nnz, f) != (size_t)nnz) {
        fclose(f);
        free(cp);
        return false;
    }
    for (i = 0; i <= n; ++i) A->ptr[i] = cp[i];
    free(cp);
    fclose(f);
    return true;
}

typedef struct {
    int col, row;
    double v;
} Trip;

static int trip_cmp(const void *a, const void *b) {
    const Trip *x = a, *y = b;
    if (x->col != y->col) return x->col < y->col ? -1 : 1;
    if (x->row != y->row) return x->row < y->row ? -1 : 1;
    return 0;
}

/* Matrix Market coordinate real|integer, symmetric|general. Lower triangle
 * only (upper entries mirrored for symmetric, dropped for general), like
 * smf::read_matrix_market; duplicates are summed. */
static bool load_mtx(const char *path, Csc *A, char *err, size_t errlen) {
    FILE *f = fopen(path, "r");
    char line[1024];
    bool symmetric = false;
    int rows = 0, cols = 0;
    long long nnz_decl = 0, k;
    Trip *t;
    int64_t m = 0, out = 0;
    int j;
    if (!f) {
        snprintf(err, errlen, "cannot open %s", path);
        return false;
    }
    if (!fgets(line, sizeof line, f) || strncmp(line, "%%MatrixMarket", 14) != 0) {
        snprintf(err, errlen, "not a Matrix Market file: %s", path);
        fclose(f);
        return false;
    }
    for (char *p = line; *p; ++p) *p = (char)tolower((unsigned char)*p);
    symmetric = strstr(line, "symmetric") != NULL;
    if (!strstr(line, "coordinate") || strstr(line, "complex")) {
        snprintf(err, errlen, "unsupported Matrix Market kind");
        fclose(f);
        return false;
    }
    do {
        if (!fgets(line, sizeof line, f)) {
            snprintf(err, errlen, "missing size line");
            fclose(f);
            return false;
        }
    } while (line[0] == '%');
    if (sscanf(line, "%d %d %lld", &rows, &cols, &nnz_decl) != 3 || rows != cols) {
        snprintf(err, errlen, "bad size line");
        fclose(f);
        return false;
    }
    t = malloc((size_t)(nnz_decl ? nnz_decl : 1) * sizeof(Trip));
    for (k = 0; k < nnz_decl; ++k) {
        int r, c;
        double v;
        if (!fgets(line, sizeof line, f) || sscanf(line, "%d %d %lf", &r, &c, &v) != 3) {
            snprintf(err, errlen, "bad entry %lld", k);
            free(t);
            fclose(f);
            return false;
        }
        --r;
        --c;
        if (r < c) {
            if (!symmetric) continue;
            int tmp = r;
            r = c;
            c = tmp;
        }
        t[m].row = r;
        t[m].col = c;
        t[m].v = v;
        ++m;
    }
    fclose(f);
    qsort(t, (size_t)m, sizeof(Trip), trip_cmp);
    A->n = rows;
    A->ptr = calloc((size_t)rows + 1, sizeof(int64_t));
    A->row = malloc((size_t)(m ? m : 1) * sizeof(int));
    A->val = malloc((size_t)(m ? m : 1) * sizeof(double));
    for (int64_t i = 0; i < m; ++i) {
        if (out > 0 && t[i].col == t[i - 1].col && t[i].row == t[i - 1].row) {
            A->val[out - 1] += t[i].v;
        } else {
            A->row[out] = t[i].row;
            A->val[out] = t[i].v;
            A->ptr[t[i].col + 1]++;
            ++out;
        }
    }
    for (j = 0; j < rows; ++j) A->ptr[j + 1] += A->ptr[j];
    A->nnz = out;
    free(t);
    return true;
}

static void save_bin_cache(const char *mtx, const Csc *A) {
    char path[4096], tmp[4100];
    FILE *f;
    int64_t n = A->n, nnz = A->nnz, i;
    int32_t *cp;
    snprintf(path, sizeof path, "%s.smfbin", mtx);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    f = fopen(tmp, "wb");
    if (!f) return;
    cp = malloc((size_t)(n + 1) * sizeof(int32_t));
    for (i = 0; i <= n; ++i) cp[i] = (int32_t)A->ptr[i];
    fwrite(kBinMagic, 1, 8, f);
    fwrite(&n, 8, 1, f);
    fwrite(&nnz, 8, 1, f);
    fwrite(cp, sizeof(int32_t), (size_t)(n + 1), f);
    fwrite(A->row, sizeof(int32_t), (size_t)nnz, f);
    fwrite(A->val, sizeof(double), (size_t)nnz, f);
    free(cp);
    if (fclose(f) == 0) rename(tmp, path);
}

/* ------------------------------------------------------------ validation */

typedef struct {
    bool finite;
    double eta_inf, omega, rel_res;
} Valid;

/* Same metrics as smf::bench::validate (worst over right-hand sides): eta_inf
 * = ||r||inf / (||A||inf ||x||inf + ||b||inf), omega = max_i |r_i| /
 * ((|A||x|)_i + |b_i|), relative_residual in the 2-norm / Frobenius form. The
 * sums are in long double (the harness uses Neumaier-compensated ones). */
static Valid validate(const Csc *A, const double *vals, const double *b, const double *x, int nrhs) {
    const int n = A->n;
    Valid agg = {true, 0.0, 0.0, 0.0};
    long double *rowsum = calloc((size_t)n, sizeof(long double));
    long double *ax = malloc((size_t)n * sizeof(long double));
    long double *aax = malloc((size_t)n * sizeof(long double));
    long double anorm = 0.0L, fro2 = 0.0L;
    for (int j = 0; j < n; ++j) {
        for (int64_t p = A->ptr[j]; p < A->ptr[j + 1]; ++p) {
            const int i = A->row[p];
            const long double a = fabsl((long double)vals[p]);
            rowsum[i] += a;
            if (i != j) rowsum[j] += a;
            fro2 += (i == j ? 1.0L : 2.0L) * a * a;
        }
    }
    for (int i = 0; i < n; ++i) anorm = rowsum[i] > anorm ? rowsum[i] : anorm;
    for (int r = 0; r < nrhs; ++r) {
        const double *bc = b + (size_t)r * n, *xc = x + (size_t)r * n;
        long double bn = 0.0L, xn = 0.0L, bn2 = 0.0L, xn2 = 0.0L, rn = 0.0L, rn2 = 0.0L, om = 0.0L;
        bool fin = true;
        for (int i = 0; i < n; ++i) {
            if (!isfinite(bc[i]) || !isfinite(xc[i])) {
                fin = false;
                break;
            }
            bn = fmaxl(bn, fabsl((long double)bc[i]));
            xn = fmaxl(xn, fabsl((long double)xc[i]));
            bn2 += (long double)bc[i] * bc[i];
            xn2 += (long double)xc[i] * xc[i];
        }
        if (!fin) {
            agg.finite = false;
            agg.eta_inf = agg.omega = agg.rel_res = INFINITY;
            continue;
        }
        memset(ax, 0, (size_t)n * sizeof(long double));
        memset(aax, 0, (size_t)n * sizeof(long double));
        for (int j = 0; j < n; ++j) {
            for (int64_t p = A->ptr[j]; p < A->ptr[j + 1]; ++p) {
                const int i = A->row[p];
                const long double v = vals[p];
                ax[i] += v * xc[j];
                aax[i] += fabsl(v) * fabsl((long double)xc[j]);
                if (i != j) {
                    ax[j] += v * xc[i];
                    aax[j] += fabsl(v) * fabsl((long double)xc[i]);
                }
            }
        }
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            const long double res = (long double)bc[i] - ax[i];
            const long double ar = fabsl(res);
            const long double den = aax[i] + fabsl((long double)bc[i]);
            rn = fmaxl(rn, ar);
            rn2 += res * res;
            if (den > 0.0L) om = fmaxl(om, ar / den);
            else if (ar != 0.0L) {
                om = INFINITY;
                ok = false;
            }
        }
        {
            const long double ed = anorm * xn + bn;
            long double eta = ed > 0.0L ? rn / ed : (rn == 0.0L ? 0.0L : INFINITY);
            const long double rd = sqrtl(fro2) * sqrtl(xn2) + sqrtl(bn2);
            long double rel = rd > 0.0L ? sqrtl(rn2) / rd : (sqrtl(rn2) == 0.0L ? 0.0L : INFINITY);
            if (!isfinite((double)eta) || !isfinite((double)om) || !isfinite((double)rel)) ok = false;
            if (!ok) agg.finite = false;
            if ((double)eta > agg.eta_inf) agg.eta_inf = (double)eta;
            if ((double)om > agg.omega) agg.omega = (double)om;
            if ((double)rel > agg.rel_res) agg.rel_res = (double)rel;
        }
    }
    free(rowsum);
    free(ax);
    free(aax);
    return agg;
}

static bool qualified(const Valid *v) {
    return v->finite && isfinite(v->eta_inf) && isfinite(v->omega) && v->eta_inf <= 1e-10 && v->omega <= 1e-8;
}

/* ---------------------------------------------------- iterative refinement */

#define REFINE_ETA_DONE 1e-14
#define REFINE_OMEGA_DONE 1e-10

typedef struct {
    const Csc *A;
    void *akeep, *fkeep;
    struct spral_ssids_options *opt;
    struct spral_ssids_inform *inf;
} Fact;

static bool ssids_solve(Fact *F, double *x, int nrhs) {
    spral_ssids_solve(0, nrhs, x, F->A->n, F->akeep, F->fkeep, F->opt, F->inf);
    return F->inf->flag >= 0;
}

/* Port of smf::bench::refine_solution (replay_refine.hpp). */
static void refine_solution(Fact *F, const double *vals, const double *b, double *x, int max_steps) {
    const Csc *A = F->A;
    const int n = A->n;
    double *ax = malloc((size_t)n * sizeof(double)), *aax = malloc((size_t)n * sizeof(double)),
           *r = malloc((size_t)n * sizeof(double)), *rowsum = calloc((size_t)n, sizeof(double));
    double anorm = 0.0, bnorm = 0.0, prev = INFINITY;
    for (int j = 0; j < n; ++j) {
        for (int64_t p = A->ptr[j]; p < A->ptr[j + 1]; ++p) {
            const int i = A->row[p];
            const double a = fabs(vals[p]);
            rowsum[i] += a;
            if (i != j) rowsum[j] += a;
        }
    }
    for (int i = 0; i < n; ++i) {
        anorm = fmax(anorm, rowsum[i]);
        bnorm = fmax(bnorm, fabs(b[i]));
    }
    for (int it = 0; it <= max_steps; ++it) {
        double rnorm = 0.0, xnorm = 0.0, omega = 0.0, eta, merit;
        memset(aax, 0, (size_t)n * sizeof(double));
        memset(ax, 0, (size_t)n * sizeof(double));
        for (int j = 0; j < n; ++j) {
            const double xj = x[j];
            double ajx = 0.0;
            for (int64_t p = A->ptr[j]; p < A->ptr[j + 1]; ++p) {
                const int i = A->row[p];
                const double a = fabs(vals[p]);
                aax[i] += a * fabs(xj);
                ax[i] += vals[p] * xj;
                if (i != j) {
                    ajx += a * fabs(x[i]);
                    ax[j] += vals[p] * x[i];
                }
            }
            aax[j] += ajx;
        }
        for (int i = 0; i < n; ++i) {
            const double den = aax[i] + fabs(b[i]);
            r[i] = b[i] - ax[i];
            rnorm = fmax(rnorm, fabs(r[i]));
            xnorm = fmax(xnorm, fabs(x[i]));
            if (den > 0.0) omega = fmax(omega, fabs(r[i]) / den);
            else if (r[i] != 0.0) omega = INFINITY;
        }
        eta = rnorm / (anorm * xnorm + bnorm);
        merit = fmax(eta / REFINE_ETA_DONE, omega / REFINE_OMEGA_DONE);
        if (it == max_steps || merit <= 1.0 || merit >= 0.5 * prev) break;
        prev = merit;
        if (!ssids_solve(F, r, 1)) break;
        for (int i = 0; i < n; ++i) x[i] += r[i];
    }
    free(ax);
    free(aax);
    free(r);
    free(rowsum);
}

/* ------------------------------------------------------------------ record */

typedef struct {
    char outcome[16];
    char reason[256];
    double analyse_ms, factor_ms, first_factor_ms, refactor_ms, solve_ms, refine_ms, wall_ms;
    int inertia_pos, inertia_neg, inertia_zero, delayed;
    int64_t factor_entries;
    double flops;
    Valid valid;
    Valid *per_rhs;
    int n_per_rhs;
    Valid *per_update;
    int n_per_update;
} Result;

static void jnum(FILE *f, double v) {
    if (!isfinite(v)) fputs("null", f);
    else fprintf(f, "%.17g", v);
}

static void jvalid(FILE *f, const Valid *v) {
    fprintf(f, "{\"finite\":%s,\"eta_inf\":", v->finite ? "true" : "false");
    jnum(f, v->eta_inf);
    fputs(",\"omega\":", f);
    jnum(f, v->omega);
    fputs(",\"relative_residual\":", f);
    jnum(f, v->rel_res);
    fputc('}', f);
}

static void jstr(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') fprintf(f, "\\%c", *s);
        else if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", (unsigned)*s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

typedef struct {
    const char *matrix_id, *sequence_id, *family_id, *mode, *track, *version;
    int threads, n, nrhs, trial_id, batch_count;
    int64_t nnz;
    bool warmup;
} Meta;

static void write_record(const char *path, const Meta *m, const Result *r) {
    FILE *f = path && *path ? fopen(path, "a") : NULL;
    if (!f) return;
    fprintf(f, "{\"schema_version\":1,\"matrix_id\":");
    jstr(f, m->matrix_id);
    fputs(",\"sequence_id\":", f);
    jstr(f, m->sequence_id);
    fputs(",\"family_id\":", f);
    jstr(f, m->family_id);
    fputs(",\"matrix_sha256\":\"\",\"rhs_sha256\":\"\",\"solver_id\":\"ssids\",\"solver_version\":", f);
    jstr(f, m->version);
    fputs(",\"library_path\":\"libspral.so (WSL2 Ubuntu)\",\"blas_backend\":\"openblas\",\"mode\":", f);
    jstr(f, m->mode);
    fputs(",\"track\":", f);
    jstr(f, m->track);
    fprintf(f, ",\"threads\":%d,\"blas_threads\":%d,\"n\":%d,\"nnz\":%lld,\"nrhs\":%d,\"trial_id\":%d,\"warmup\":%s,"
               "\"batch_count\":%d,\"outcome\":\"%s\",\"reason\":",
            m->threads, m->threads, m->n, (long long)m->nnz, m->nrhs, m->trial_id, m->warmup ? "true" : "false",
            m->batch_count, r->outcome);
    jstr(f, r->reason);
    fputs(",\"analyse_ms\":", f);
    jnum(f, r->analyse_ms);
    fputs(",\"factor_ms\":", f);
    jnum(f, r->factor_ms);
    if (r->first_factor_ms >= 0) {
        fputs(",\"first_factor_ms\":", f);
        jnum(f, r->first_factor_ms);
    }
    if (r->refactor_ms >= 0) {
        fputs(",\"refactor_ms\":", f);
        jnum(f, r->refactor_ms);
    }
    fputs(",\"solve_ms\":", f);
    jnum(f, r->solve_ms);
    fputs(",\"refine_ms\":", f);
    jnum(f, r->refine_ms);
    fputs(",\"wall_ms\":", f);
    jnum(f, r->wall_ms);
    fprintf(f, ",\"peak_rss_bytes\":%lld,\"inertia\":{\"pos\":%d,\"neg\":%d,\"zero\":%d},\"delayed_pivots\":%d,"
               "\"factor_entries\":%lld,\"flops\":",
            (long long)peak_rss_bytes(), r->inertia_pos, r->inertia_neg, r->inertia_zero, r->delayed,
            (long long)r->factor_entries);
    jnum(f, r->flops);
    fputs(",\"validation\":", f);
    jvalid(f, &r->valid);
    fputs(",\"per_rhs_validation\":[", f);
    for (int i = 0; i < r->n_per_rhs; ++i) {
        if (i) fputc(',', f);
        jvalid(f, &r->per_rhs[i]);
    }
    fputs("],\"per_update_validation\":[", f);
    for (int i = 0; i < r->n_per_update; ++i) {
        if (i) fputc(',', f);
        jvalid(f, &r->per_update[i]);
    }
    fputs("]}\n", f);
    fclose(f);
}

/* ------------------------------------------------------------------- main */

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void set_outcome(Result *r, const char *o, const char *why) {
    snprintf(r->outcome, sizeof r->outcome, "%s", o);
    snprintf(r->reason, sizeof r->reason, "%s", why);
}

/* Factor outcome from the SSIDS flag. */
static const char *factor_outcome(int flag, char *why, size_t len) {
    if (flag >= 0) return "Success";
    snprintf(why, len, "ssids_factor flag=%d", flag);
    if (flag == -5) return "Singular";
    if (flag == -50) return "OutOfMemory";
    return "Failed";
}

static bool do_factor(Fact *F, bool posdef, const double *vals, Result *r, double *ms) {
    char why[128];
    const double t0 = now_ms();
    spral_ssids_factor(posdef, F->A->ptr, F->A->row, vals, NULL, F->akeep, &F->fkeep, F->opt, F->inf);
    *ms = now_ms() - t0;
    {
        const char *o = factor_outcome(F->inf->flag, why, sizeof why);
        if (strcmp(o, "Success") != 0) {
            set_outcome(r, o, why);
            return false;
        }
    }
    r->inertia_neg = posdef ? 0 : F->inf->num_neg;
    r->inertia_zero = posdef ? 0 : F->A->n - F->inf->matrix_rank;
    r->inertia_pos = F->A->n - r->inertia_neg - r->inertia_zero;
    r->delayed = F->inf->num_delay;
    r->factor_entries = F->inf->num_factor;
    r->flops = (double)F->inf->num_flops;
    return true;
}

static bool do_analyse(Fact *F, const double *vals, Result *r, double *ms) {
    double t0;
    /* Freeing the previous analysis is not part of the analyse phase. */
    if (F->akeep || F->fkeep) spral_ssids_free(&F->akeep, &F->fkeep);
    F->akeep = F->fkeep = NULL;
    t0 = now_ms();
    spral_ssids_analyse(false, F->A->n, NULL, F->A->ptr, F->A->row, vals, &F->akeep, F->opt, F->inf);
    *ms = now_ms() - t0;
    if (F->inf->flag < 0) {
        char why[64];
        snprintf(why, sizeof why, "ssids_analyse flag=%d", F->inf->flag);
        set_outcome(r, F->inf->flag == -50 ? "OutOfMemory" : "Failed", why);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const char *manifest = "", *case_id = "ssids_case", *sequence_id = NULL, *family_id = NULL, *mode = "cold",
               *track = "recommended", *jsonl = "", *mtx = NULL, *mtype = "indefinite", *seq_path = NULL;
    int trial = 0, nrhs = 1, threads = 1, refine_steps = 3, seq_steps = 5;
    double pivot_u = -1.0;
    int ord_override = -1, scal_override = -1;
    bool warmup = false;
    Csc A = {0};
    Meta meta;
    Result res;
    Fact F;
    struct spral_ssids_options opt;
    struct spral_ssids_inform inf;
    char err[256] = "";
    bool spd;

    (void)manifest;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--warmup")) warmup = true;
        else if (!v) continue;
        else if (!strcmp(a, "--manifest")) { manifest = v; ++i; }
        else if (!strcmp(a, "--case")) { case_id = v; ++i; }
        else if (!strcmp(a, "--sequence-id")) { sequence_id = v; ++i; }
        else if (!strcmp(a, "--family-id")) { family_id = v; ++i; }
        else if (!strcmp(a, "--solver")) { ++i; }
        else if (!strcmp(a, "--mode")) { mode = v; ++i; }
        else if (!strcmp(a, "--track")) { track = v; ++i; }
        else if (!strcmp(a, "--trial")) { trial = atoi(v); ++i; }
        else if (!strcmp(a, "--nrhs")) { nrhs = atoi(v) > 0 ? atoi(v) : 1; ++i; }
        else if (!strcmp(a, "--threads")) { threads = atoi(v) > 0 ? atoi(v) : 1; ++i; }
        else if (!strcmp(a, "--mtx")) { mtx = v; ++i; }
        else if (!strcmp(a, "--mtype")) { mtype = v; ++i; }
        else if (!strcmp(a, "--seq-values")) { seq_path = v; ++i; }
        else if (!strcmp(a, "--JsonlOut")) { jsonl = v; ++i; }
        else if (!strcmp(a, "--pivot-u")) { pivot_u = atof(v); ++i; }
        else if (!strcmp(a, "--refine-steps")) { refine_steps = atoi(v) > 0 ? atoi(v) : 0; ++i; }
        else if (!strcmp(a, "--ssids-ordering")) { ord_override = atoi(v); ++i; }
        else if (!strcmp(a, "--ssids-scaling")) { scal_override = atoi(v); ++i; }
        else if (!strcmp(a, "--seq-steps")) { seq_steps = !strcmp(v, "all") ? -1 : (atoi(v) > 0 ? atoi(v) : 1); ++i; }
        else if (!strcmp(a, "--ordering") || !strcmp(a, "--scaling") || !strcmp(a, "--ipm-static")) { ++i; } /* smf-only */
    }
    if (!sequence_id) sequence_id = case_id;
    if (!family_id) family_id = sequence_id;
    spd = !strcmp(mtype, "spd");

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    memset(&meta, 0, sizeof meta);
    memset(&res, 0, sizeof res);
    res.first_factor_ms = res.refactor_ms = -1.0;
    meta.matrix_id = case_id;
    meta.sequence_id = sequence_id;
    meta.family_id = family_id;
    meta.mode = mode;
    meta.track = track;
    meta.version = "SPRAL SSIDS 2025.09.18 (CPU)";
    meta.threads = threads;
    meta.nrhs = nrhs;
    meta.trial_id = trial;
    meta.warmup = warmup;
    meta.batch_count = 1;

    if (!mtx) {
        set_outcome(&res, "Unavailable", "ssids_replay needs --mtx (run_wsl.py exports built-in cases)");
        write_record(jsonl, &meta, &res);
        return 1;
    }
    if (!load_bin_cache(mtx, &A)) {
        if (!load_mtx(mtx, &A, err, sizeof err)) {
            set_outcome(&res, "InvalidInput", err);
            write_record(jsonl, &meta, &res);
            return 1;
        }
        save_bin_cache(mtx, &A);
    }
    meta.n = A.n;
    meta.nnz = A.nnz;

    spral_ssids_default_options(&opt);
    opt.print_level = -1;
    opt.u = pivot_u >= 0.0 ? pivot_u : (!strcmp(track, "ipopt") ? 1e-8 : 0.01);
    if (spd) {
        opt.ordering = 1;
        opt.scaling = 0;
    } else {
        opt.ordering = 2;
        opt.scaling = 3;
    }
    if (ord_override >= 0) opt.ordering = ord_override;
    if (scal_override >= 0) opt.scaling = scal_override;
    memset(&inf, 0, sizeof inf);
    F.A = &A;
    F.akeep = F.fkeep = NULL;
    F.opt = &opt;
    F.inf = &inf;

    const size_t n = (size_t)A.n, total = n * (size_t)nrhs;
    double *b = malloc(total * sizeof(double)), *x = malloc(total * sizeof(double));
    for (size_t i = 0; i < total; ++i) b[i] = 1.0;
    memcpy(x, b, total * sizeof(double));

    /* Untimed in-process warm-up (one-time runtime initialisation). */
    {
        Result tmp;
        double ms;
        double *xw = malloc(total * sizeof(double));
        memcpy(xw, b, total * sizeof(double));
        memset(&tmp, 0, sizeof tmp);
        if (do_analyse(&F, A.val, &tmp, &ms) && do_factor(&F, spd, A.val, &tmp, &ms)) (void)ssids_solve(&F, xw, nrhs);
        free(xw);
        memset(&res, 0, sizeof res);
        res.first_factor_ms = res.refactor_ms = -1.0;
    }

    bool ok = true;
    if (!strcmp(mode, "warm_sequence")) {
        int nsteps_k = 5;
        size_t *picks = NULL;
        size_t step_bytes = (size_t)A.nnz * sizeof(double);
        FILE *sf = NULL;
        bool recorded = false;
        if (seq_path) {
            size_t nstored;
            sf = fopen(seq_path, "rb");
            if (!sf) {
                set_outcome(&res, "InvalidInput", "unreadable --seq-values file");
                write_record(jsonl, &meta, &res);
                return 1;
            }
            fseek(sf, 0, SEEK_END);
            nstored = step_bytes ? (size_t)ftell(sf) / step_bytes : 0;
            if (nstored == 0) {
                set_outcome(&res, "InvalidInput", "unreadable --seq-values file");
                write_record(jsonl, &meta, &res);
                return 1;
            }
            nsteps_k = seq_steps < 0 ? (int)nstored : seq_steps;
            picks = malloc((size_t)nsteps_k * sizeof(size_t));
            for (int s = 0; s < nsteps_k; ++s) {
                if (seq_steps < 0) picks[s] = (size_t)s;
                else
                    picks[s] = (nstored == 1 || nsteps_k == 1) ? 0 : (size_t)s * (nstored - 1) / (size_t)(nsteps_k - 1);
            }
        }
        meta.batch_count = nsteps_k;
        double **sv = malloc((size_t)nsteps_k * sizeof(double *));
        double **sol = malloc((size_t)nsteps_k * sizeof(double *));
        for (int s = 0; s < nsteps_k; ++s) {
            sv[s] = malloc(step_bytes ? step_bytes : 8);
            sol[s] = malloc(total * sizeof(double));
            memcpy(sol[s], b, total * sizeof(double));
            memcpy(sv[s], A.val, step_bytes);
        }
        if (sf) {
            recorded = true;
            for (int s = 0; s < nsteps_k; ++s) {
                fseek(sf, (long)(picks[s] * step_bytes), SEEK_SET);
                if (fread(sv[s], 1, step_bytes, sf) != step_bytes) recorded = false;
            }
            fclose(sf);
        }
        for (int s = 0; s < nsteps_k && !recorded; ++s) {
            const double delta = 1.0 + 0.05 * (double)s;
            for (int j = 0; j < A.n; ++j) {
                const int64_t p0 = A.ptr[j];
                if (p0 < A.ptr[j + 1] && A.row[p0] == j && A.val[p0] != 0.0) sv[s][p0] = A.val[p0] * delta;
            }
        }

        const bool refine = !strcmp(track, "ipopt") && nrhs == 1;
        double *step_f = malloc((size_t)nsteps_k * sizeof(double));
        int nf = 0, last = nsteps_k - 1;
        const double t0 = now_ms();
        ok = do_analyse(&F, A.val, &res, &res.analyse_ms);
        if (ok) {
            for (int s = 0; s < nsteps_k; ++s) {
                double fms = 0.0;
                bool fok = do_factor(&F, spd, sv[s], &res, &fms);
                res.factor_ms += fms;
                if (!fok) {
                    ok = false;
                    break;
                }
                step_f[nf++] = fms;
                const double ts = now_ms();
                if (!ssids_solve(&F, sol[s], nrhs)) {
                    set_outcome(&res, "Failed", "ssids_solve failed");
                    ok = false;
                    break;
                }
                res.solve_ms += now_ms() - ts;
                if (refine) {
                    const double tr = now_ms();
                    refine_solution(&F, sv[s], b, sol[s], refine_steps);
                    res.refine_ms += now_ms() - tr;
                }
            }
        }
        res.wall_ms = now_ms() - t0;
        if (nf > 0) {
            res.first_factor_ms = step_f[0];
            if (nf > 1) {
                qsort(step_f + 1, (size_t)(nf - 1), sizeof(double), cmp_double);
                const int m = nf - 1;
                res.refactor_ms = (m % 2) ? step_f[1 + m / 2] : 0.5 * (step_f[m / 2] + step_f[1 + m / 2]);
            }
        }
        if (ok) {
            res.per_update = malloc((size_t)nsteps_k * sizeof(Valid));
            res.n_per_update = nsteps_k;
            bool all_q = true;
            for (int s = 0; s < nsteps_k; ++s) {
                res.per_update[s] = validate(&A, sv[s], b, sol[s], nrhs);
                if (!qualified(&res.per_update[s])) all_q = false;
            }
            res.per_rhs = malloc((size_t)nrhs * sizeof(Valid));
            res.n_per_rhs = nrhs;
            for (int q = 0; q < nrhs; ++q) res.per_rhs[q] = validate(&A, sv[last], b + (size_t)q * n, sol[last] + (size_t)q * n, 1);
            res.valid = validate(&A, sv[last], b, sol[last], nrhs);
            if (qualified(&res.valid) && all_q) set_outcome(&res, "Success", "");
            else set_outcome(&res, "Inaccurate", "exceeded backward error threshold");
        }
    } else {
        const int batch = (A.n <= 64) ? 10 : 1;
        double sa = 0, sf = 0, ss = 0;
        meta.batch_count = batch;
        const double t0 = now_ms();
        for (int it = 0; it < batch && ok; ++it) {
            double ms = 0.0;
            memcpy(x, b, total * sizeof(double));
            ok = do_analyse(&F, A.val, &res, &ms);
            sa += ms;
            if (ok) {
                ok = do_factor(&F, spd, A.val, &res, &ms);
                sf += ms;
            }
            if (ok) {
                const double ts = now_ms();
                ok = ssids_solve(&F, x, nrhs);
                if (!ok) set_outcome(&res, "Failed", "ssids_solve failed");
                ss += now_ms() - ts;
            }
        }
        res.wall_ms = (now_ms() - t0) / batch;
        res.analyse_ms = sa / batch;
        res.factor_ms = sf / batch;
        res.solve_ms = ss / batch;
        if (ok) {
            res.per_rhs = malloc((size_t)nrhs * sizeof(Valid));
            res.n_per_rhs = nrhs;
            for (int q = 0; q < nrhs; ++q) res.per_rhs[q] = validate(&A, A.val, b + (size_t)q * n, x + (size_t)q * n, 1);
            res.valid = validate(&A, A.val, b, x, nrhs);
            if (qualified(&res.valid)) set_outcome(&res, "Success", "");
            else set_outcome(&res, "Inaccurate", "exceeded backward error threshold");
        }
    }
    write_record(jsonl, &meta, &res);
    return strcmp(res.outcome, "Success") == 0 ? 0 : 1;
}
