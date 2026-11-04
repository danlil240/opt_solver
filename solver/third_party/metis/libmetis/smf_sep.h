/* smf patch: METIS_ComputeVertexSeparator split into stages, so that the
   independent L1 runs of MlevelNodeBisectionL2 can run as separate tasks
   (smf's parallel nested dissection, ordering_parallel_nd.cpp). See
   smf_sep.c. */
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

/* Sets up the separator of the graph (xadj, adjncy, vwgt as for
   METIS_ComputeVertexSeparator; they must stay valid and unchanged until
   smf_sep_finish() or smf_sep_free()) with RNG seed `seed`, and coarsens it
   as MlevelNodeBisectionL2 does. *nruns: the runs to do with smf_sep_run()
   (SMF_SEP_NRUNS), or 0 when the graph is small enough for one direct run,
   which smf_sep_begin() then did. Returns METIS_OK and the handle in *out,
   or an error code (nothing allocated). */
int smf_sep_begin(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                  smf_sep_t **out, int *nruns);

/* Run i in [0, nruns): one MlevelNodeBisectionL1 on the coarse graph, on its
   own control copy and workspace, seeded from (seed, i). Different i may run
   concurrently on different threads; the result depends on (seed, i) only.
   Returns METIS_OK or METIS_ERROR_MEMORY. */
int smf_sep_run(smf_sep_t *s, int i);

/* After every run: keeps the first run with the smallest separator, refines
   it back to the input graph (reseeded from seed), writes part[] (0, 1 or 2
   per vertex) and *sepsize, and frees the handle. Returns METIS_OK or an
   error code (handle freed either way). */
int smf_sep_finish(smf_sep_t *s, idx_t *part, idx_t *sepsize);

/* Frees a handle without finishing it (null allowed). */
void smf_sep_free(smf_sep_t *s);

#ifdef __cplusplus
}
#endif
#endif
