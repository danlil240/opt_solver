/* smf patch: METIS_ComputeVertexSeparator split into stages, plus fast leaf
   ordering and parallel coarsening hooks for smf's parallel nested dissection
   (ordering_parallel_nd.cpp). See smf_sep.c. */
#ifndef SMF_SEP_H
#define SMF_SEP_H

#include "metis.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct smf_sep_s smf_sep_t;

/* Number of runs smf_sep_begin() leaves for smf_sep_run() on a graph past
   L2's single-run threshold (MlevelNodeBisectionL2's nruns). */
#define SMF_SEP_NRUNS 5

/* Callback for parallel chunk execution inside CreateCoarseGraph (bitwise
   identical to serial CreateCoarseGraph). */
typedef void (*smf_pfor_fn)(int ntasks, void (*body)(int, void *), void *ctx, void *pool);

typedef struct {
  int nruns;              /* <= 0: SMF_SEP_NRUNS (5); 1: direct L1 run */
  int niter;              /* <= 0: METIS default (10) */
  int niparts;            /* <= 0: METIS default (7) */
  int no2hop;             /* 0 or 1 */
  int compressed;         /* 1 if graph was compressed, 0 otherwise */
  smf_pfor_fn pfor;       /* optional parallel-for callback for contraction */
  void *pfor_pool;
  int pfor_threads;       /* max threads to use in contraction */
  volatile int *cancel;   /* optional cancellation flag (non-zero => abort) */
} smf_sep_opts_t;

/* Sets up the separator of the graph with default METIS options (equivalent to
   smf_sep_begin_ex with NULL opts). */
int smf_sep_begin(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                  smf_sep_t **out, int *nruns);

/* Configurable variant of smf_sep_begin(). */
int smf_sep_begin_ex(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                     const smf_sep_opts_t *opts, smf_sep_t **out, int *nruns);

/* Run i in [0, nruns): one MlevelNodeBisectionL1 on the coarse graph, on its
   own control copy and workspace, seeded from (seed, i). Different i may run
   concurrently on different threads; the result depends on (seed, i) only.
   Returns METIS_OK or an error code. */
int smf_sep_run(smf_sep_t *s, int i);

/* After every run: keeps the first run with the smallest separator, refines
   it back to the input graph (reseeded from seed), writes part[] (0, 1 or 2
   per vertex) and *sepsize, and frees the handle. Returns METIS_OK or an
   error code (handle freed either way). */
int smf_sep_finish(smf_sep_t *s, idx_t *part, idx_t *sepsize);

/* Frees a handle without finishing it (null allowed). */
void smf_sep_free(smf_sep_t *s);

/* Fast METIS nested dissection on a leaf subgraph with one shared ctrl_t /
   workspace, optional weight-based MMD switch, and cancellation support.
   perm[new] = old, iperm[old] = new. */
int smf_nd_leaf(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                const smf_sep_opts_t *opts, idx_t mmd_switch_wgt,
                idx_t *perm, idx_t *iperm);

#ifdef __cplusplus
}
#endif
#endif
