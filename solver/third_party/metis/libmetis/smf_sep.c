/* smf patch: METIS_ComputeVertexSeparator in stages (smf_sep.h).

   METIS_ComputeVertexSeparator -> MlevelNodeBisectionMultiple (nseps = 1)
   -> MlevelNodeBisectionL2: coarsen 4 levels, then SMF_SEP_NRUNS
   independent MlevelNodeBisectionL1 runs on the coarse graph, keep the first
   smallest separator, and refine it back (Refine2WayNode). The runs share
   only the RNG stream and the workspace of one ctrl_t. Here each run gets
   its own ctrl_t copy, workspace and graph view of the (read-only) coarse
   graph, and its own seed, so the runs can be separate tasks and the result
   depends on the seed only, not on which thread runs what. It is not
   bitwise METIS_ComputeVertexSeparator (whose runs continue one RNG
   stream), but the same algorithm. */
#include "metislib.h"
#include "smf_sep.h"

struct smf_sep_s {
  ctrl_t *ctrl;
  graph_t *graph;     /* the input graph (caller's arrays) */
  graph_t *cgraph;    /* its 4-level coarsening (graph->coarser chain); NULL: direct run done */
  idx_t seed;
  int nruns;
  idx_t *wheres;      /* nruns x cgraph->nvtxs */
  idx_t *mincuts;
  int *done;
};

/* The coarsening chain below graph (graph itself is the caller's to free). */
static void smf_free_coarser(graph_t *graph)
{
  graph_t *g = graph->coarser, *next;
  graph->coarser = NULL;
  while (g != NULL) {
    next = g->coarser;
    FreeGraph(&g);
    g = next;
  }
}

void smf_sep_free(smf_sep_t *s)
{
  if (s == NULL)
    return;
  if (s->graph != NULL) {
    smf_free_coarser(s->graph);
    FreeGraph(&s->graph);
  }
  if (s->ctrl != NULL)
    FreeCtrl(&s->ctrl);
  gk_free((void **)&s->wheres, &s->mincuts, &s->done, LTERM);
  gk_free((void **)&s, LTERM);
}

int smf_sep_begin(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                  smf_sep_t **out, int *nruns)
{
  idx_t options[METIS_NOPTIONS];
  smf_sep_t *s;
  ctrl_t *ctrl;
  graph_t *graph;

  *out = NULL;
  *nruns = 0;
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_SEED] = seed;
  options[METIS_OPTION_NUMBERING] = 0;

  /* As METIS_ComputeVertexSeparator. */
  if ((ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL)) == NULL)
    return METIS_ERROR_INPUT;
  InitRandom(ctrl->seed);
  graph = SetupGraph(ctrl, nvtxs, 1, xadj, adjncy, vwgt, NULL, NULL);
  AllocateWorkSpace(ctrl, graph);
  ctrl->CoarsenTo = 100;

  s = (smf_sep_t *)gk_malloc(sizeof(smf_sep_t), "smf_sep_begin: s");
  memset(s, 0, sizeof(smf_sep_t));
  s->ctrl = ctrl;
  s->graph = graph;
  s->seed = ctrl->seed;

  if (ctrl->nseps != 1 || graph->nvtxs < 5000) {
    /* MlevelNodeBisectionMultiple as it is: L2's single direct run here
       (nseps is 1 with smf's options). */
    MlevelNodeBisectionMultiple(ctrl, graph);
    *out = s;
    return METIS_OK;
  }

  /* MlevelNodeBisectionL2 up to its runs. */
  WCOREPUSH;
  ctrl->CoarsenTo = gk_max(100, graph->nvtxs/30);
  s->cgraph = CoarsenGraphNlevels(ctrl, graph, 4);
  WCOREPOP;

  s->nruns = SMF_SEP_NRUNS;
  s->wheres = imalloc((size_t)s->nruns * (size_t)s->cgraph->nvtxs, "smf_sep_begin: wheres");
  s->mincuts = imalloc(s->nruns, "smf_sep_begin: mincuts");
  s->done = (int *)gk_malloc(sizeof(int) * s->nruns, "smf_sep_begin: done");
  memset(s->done, 0, sizeof(int) * s->nruns);
  *nruns = s->nruns;
  *out = s;
  return METIS_OK;
}

int smf_sep_run(smf_sep_t *s, int i)
{
  graph_t *cg = s->cgraph, *g;
  ctrl_t c;
  idx_t maxvwgt[1] = {0};
  real_t pijbm[3];  /* nparts (3) x ncon (1) */

  if (cg == NULL || i < 0 || i >= s->nruns)
    return METIS_ERROR_INPUT;

  /* A private control: its own workspace and neighbour pools, and the
     maxvwgt[] that coarsening writes and pijbm[] that InitSeparator's
     Setup2WayBalMultipliers writes. Everything else in it is read only. */
  c = *s->ctrl;
  c.mcore = NULL;
  c.cnbrpool = NULL;
  c.vnbrpool = NULL;
  c.nbrpoolsize = c.nbrpoolcpos = c.nbrpoolsize_max = c.nbrpoolreallocs = 0;
  c.maxvwgt = maxvwgt;
  memcpy(pijbm, s->ctrl->pijbm, sizeof(pijbm));
  c.pijbm = pijbm;

  /* A view of the coarse graph: L1 coarsens and partitions the view, which
     owns only what L1 adds to it. */
  g = SetupGraph(&c, cg->nvtxs, 1, cg->xadj, cg->adjncy, cg->vwgt, NULL, cg->adjwgt);
  AllocateWorkSpace(&c, g);
  InitRandom(s->seed + 1000 + i);
  MlevelNodeBisectionL1(&c, g, 0.7*LARGENIPARTS);

  icopy(cg->nvtxs, g->where, s->wheres + (size_t)i * (size_t)cg->nvtxs);
  s->mincuts[i] = g->mincut;
  s->done[i] = 1;

  FreeGraph(&g);
  FreeWorkSpace(&c);
  return METIS_OK;
}

int smf_sep_finish(smf_sep_t *s, idx_t *part, idx_t *sepsize)
{
  int i, best = 0;
  graph_t *graph = s->graph, *cg = s->cgraph;

  if (cg != NULL) {
    for (i = 0; i < s->nruns; i++) {
      if (!s->done[i]) {
        smf_sep_free(s);
        return METIS_ERROR;
      }
    }
    /* The first run with the smallest separator (L2's choice). */
    for (i = 1; i < s->nruns; i++)
      if (s->mincuts[i] < s->mincuts[best])
        best = i;
    Allocate2WayNodePartitionMemory(s->ctrl, cg);
    icopy(cg->nvtxs, s->wheres + (size_t)best * (size_t)cg->nvtxs, cg->where);
    cg->mincut = s->mincuts[best];

    /* L2's refinement back to the input graph, on a fixed RNG stream (the
       finishing thread's own stream would otherwise decide it). */
    InitRandom(s->seed + 2000);
    Refine2WayNode(s->ctrl, graph, cg);
    s->cgraph = NULL;  /* freed by the projection */
  }

  *sepsize = graph->pwgts[2];
  icopy(graph->nvtxs, graph->where, part);
  smf_sep_free(s);
  return METIS_OK;
}
