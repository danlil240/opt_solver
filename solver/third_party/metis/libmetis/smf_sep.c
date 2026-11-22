/* smf patch: METIS_ComputeVertexSeparator in stages + fast leaf ND (smf_sep.h).

   METIS_ComputeVertexSeparator -> MlevelNodeBisectionMultiple (nseps = 1)
   -> MlevelNodeBisectionL2: coarsen 4 levels, then nruns independent
   MlevelNodeBisectionL1 runs on the coarse graph, keep the first smallest
   separator, and refine it back (Refine2WayNode). Here each run gets its own
   ctrl_t copy, workspace and graph view of the (read-only) coarse graph, and
   its own seed, so the runs can be separate tasks and the result depends on
   the seed only, not on which thread runs what. */
#include "metislib.h"
#include "smf_sep.h"

struct smf_sep_s {
  ctrl_t *ctrl;
  graph_t *graph;     /* the input graph (caller's arrays) */
  graph_t *cgraph;    /* its 4-level coarsening (graph->coarser chain); NULL: direct run done */
  idx_t seed;
  idx_t niparts;
  int nruns;
  idx_t *wheres;      /* nruns x cgraph->nvtxs */
  idx_t *mincuts;
  int *done;
  volatile int *cancel;
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
  return smf_sep_begin_ex(nvtxs, xadj, adjncy, vwgt, seed, NULL, out, nruns);
}

int smf_sep_begin_ex(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                     const smf_sep_opts_t *opts, smf_sep_t **out, int *nruns)
{
  idx_t options[METIS_NOPTIONS];
  smf_sep_t *s;
  ctrl_t *ctrl;
  graph_t *graph;
  int eff_nruns = SMF_SEP_NRUNS;
  idx_t eff_niparts = LARGENIPARTS;

  *out = NULL;
  *nruns = 0;

  if (opts && opts->cancel && *opts->cancel)
    return METIS_ERROR;

  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_SEED] = seed;
  options[METIS_OPTION_NUMBERING] = 0;
  if (opts) {
    if (opts->niter > 0)
      options[METIS_OPTION_NITER] = opts->niter;
    if (opts->no2hop)
      options[METIS_OPTION_NO2HOP] = 1;
    options[METIS_OPTION_COMPRESS] = opts->compressed ? 1 : 0;
    if (opts->nruns > 0)
      eff_nruns = opts->nruns;
    if (opts->niparts > 0)
      eff_niparts = opts->niparts;
  }

  /* As METIS_ComputeVertexSeparator. */
  if ((ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL)) == NULL)
    return METIS_ERROR_INPUT;
  if (opts) {
    ctrl->smf_l2_nruns     = eff_nruns;
    ctrl->smf_niparts      = eff_niparts;
    ctrl->smf_pfor         = opts->pfor;
    ctrl->smf_pfor_pool    = opts->pfor_pool;
    ctrl->smf_pfor_threads = opts->pfor_threads;
    ctrl->smf_cancel       = opts->cancel;
  }
  InitRandom(ctrl->seed);
  graph = SetupGraph(ctrl, nvtxs, 1, xadj, adjncy, vwgt, NULL, NULL);
  AllocateWorkSpace(ctrl, graph);
  ctrl->CoarsenTo = 100;

  s = (smf_sep_t *)gk_malloc(sizeof(smf_sep_t), "smf_sep_begin: s");
  memset(s, 0, sizeof(smf_sep_t));
  s->ctrl = ctrl;
  s->graph = graph;
  s->seed = ctrl->seed;
  s->niparts = eff_niparts;
  s->cancel = opts ? opts->cancel : NULL;

  if (ctrl->nseps != 1 || graph->nvtxs < 5000 || eff_nruns <= 1) {
    MlevelNodeBisectionMultiple(ctrl, graph);
    if (s->cancel && *s->cancel) {
      smf_sep_free(s);
      return METIS_ERROR;
    }
    *out = s;
    return METIS_OK;
  }

  /* MlevelNodeBisectionL2 up to its runs. */
  WCOREPUSH;
  ctrl->CoarsenTo = gk_max(100, graph->nvtxs/30);
  s->cgraph = CoarsenGraphNlevels(ctrl, graph, 4);
  WCOREPOP;

  if (s->cancel && *s->cancel) {
    smf_sep_free(s);
    return METIS_ERROR;
  }

  s->nruns = eff_nruns;
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
  idx_t nip;

  if (cg == NULL || i < 0 || i >= s->nruns)
    return METIS_ERROR_INPUT;
  if (s->cancel && *s->cancel)
    return METIS_ERROR;

  /* A private control: its own workspace and neighbour pools, and the
     maxvwgt[] that coarsening writes and pijbm[] that InitSeparator's
     Setup2WayBalMultipliers writes. Disable nested pfor inside concurrent runs. */
  c = *s->ctrl;
  c.mcore = NULL;
  c.cnbrpool = NULL;
  c.vnbrpool = NULL;
  c.nbrpoolsize = c.nbrpoolcpos = c.nbrpoolsize_max = c.nbrpoolreallocs = 0;
  c.maxvwgt = maxvwgt;
  c.smf_pfor = NULL;
  c.smf_pfor_pool = NULL;
  c.smf_pfor_threads = 1;
  c.smf_cancel = s->cancel;
  memcpy(pijbm, s->ctrl->pijbm, sizeof(pijbm));
  c.pijbm = pijbm;

  /* A view of the coarse graph: L1 coarsens and partitions the view, which
     owns only what L1 adds to it. */
  g = SetupGraph(&c, cg->nvtxs, 1, cg->xadj, cg->adjncy, cg->vwgt, NULL, cg->adjwgt);
  AllocateWorkSpace(&c, g);
  InitRandom(s->seed + 1000 + i);
  nip = (idx_t)(0.7 * s->niparts);
  if (nip < 1) nip = 1;
  MlevelNodeBisectionL1(&c, g, nip);

  if (s->cancel && *s->cancel) {
    FreeGraph(&g);
    FreeWorkSpace(&c);
    return METIS_ERROR;
  }

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

  if (s->cancel && *s->cancel) {
    smf_sep_free(s);
    return METIS_ERROR;
  }

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

int smf_nd_leaf(idx_t nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t seed,
                const smf_sep_opts_t *opts, idx_t mmd_switch_wgt,
                idx_t *perm, idx_t *iperm)
{
  idx_t options[METIS_NOPTIONS];
  idx_t i, ii, j, l, nnvtxs = 0;
  ctrl_t *ctrl;
  graph_t *graph = NULL;
  idx_t *cptr = NULL, *cind = NULL;

  if (opts && opts->cancel && *opts->cancel)
    return METIS_ERROR;
  if (nvtxs <= 0)
    return METIS_OK;

  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_SEED] = seed;
  options[METIS_OPTION_NUMBERING] = 0;
  if (opts) {
    if (opts->niter > 0)
      options[METIS_OPTION_NITER] = opts->niter;
    if (opts->no2hop)
      options[METIS_OPTION_NO2HOP] = 1;
    options[METIS_OPTION_COMPRESS] = opts->compressed ? 1 : 0;
  }

  if ((ctrl = SetupCtrl(METIS_OP_OMETIS, options, 1, 3, NULL, NULL)) == NULL)
    return METIS_ERROR_INPUT;

  if (opts) {
    ctrl->smf_l2_nruns   = (opts->nruns > 0) ? opts->nruns : 1;
    ctrl->smf_niparts    = opts->niparts;
    ctrl->smf_cancel     = opts->cancel;
  } else {
    ctrl->smf_l2_nruns   = 1;
  }
  ctrl->smf_mmd_switch = mmd_switch_wgt;

  InitRandom(ctrl->seed);

  if (ctrl->compress) {
    cptr = imalloc(nvtxs + 1, "smf_nd_leaf: cptr");
    cind = imalloc(nvtxs, "smf_nd_leaf: cind");
    graph = CompressGraph(ctrl, nvtxs, xadj, adjncy, vwgt, cptr, cind);
    if (graph == NULL) {
      gk_free((void **)&cptr, &cind, LTERM);
      ctrl->compress = 0;
    } else {
      nnvtxs = graph->nvtxs;
      ctrl->cfactor = 1.0 * nvtxs / nnvtxs;
    }
  }

  if (graph == NULL)
    graph = SetupGraph(ctrl, nvtxs, 1, xadj, adjncy, vwgt, NULL, NULL);

  AllocateWorkSpace(ctrl, graph);

  if (graph->nvtxs > MMDSWITCH && graph->nedges > 0 &&
      (ctrl->smf_mmd_switch <= 0 || graph->tvwgt[0] > ctrl->smf_mmd_switch)) {
    MlevelNestedDissection(ctrl, graph, iperm, graph->nvtxs);
  } else {
    MMDOrder(ctrl, graph, iperm, graph->nvtxs);
    FreeGraph(&graph);
  }

  if (ctrl->smf_cancel && *ctrl->smf_cancel) {
    gk_free((void **)&cptr, &cind, LTERM);
    FreeCtrl(&ctrl);
    return METIS_ERROR;
  }

  if (ctrl->compress) {
    for (i = 0; i < nnvtxs; i++)
      perm[iperm[i]] = i;
    for (l = ii = 0; ii < nnvtxs; ii++) {
      i = perm[ii];
      for (j = cptr[i]; j < cptr[i + 1]; j++)
        iperm[cind[j]] = l++;
    }
    gk_free((void **)&cptr, &cind, LTERM);
  }

  for (i = 0; i < nvtxs; i++)
    perm[iperm[i]] = i;

  FreeCtrl(&ctrl);
  return METIS_OK;
}
