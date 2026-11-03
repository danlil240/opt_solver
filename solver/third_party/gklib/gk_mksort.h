/*
 * gk_mksort.h -- GK_MKQSORT, GKlib's in-line sort template.
 *
 * smf patch 4: this file replaces GKlib's original gk_mksort.h, which was
 * adapted from the GNU C Library's qsort and carried its LGPL-2.1 licence.
 * Written from scratch for smf from a description of that algorithm (not
 * from its code) and licensed under the Apache License 2.0, like the rest
 * of GKlib (see LICENSE.txt and solver/third_party/README.md).
 *
 * It reproduces the original's element order, equal keys included, so
 * METIS's orderings are unchanged (smf test ParallelNd.GkSortPinnedOrder):
 *   - quicksort while a range has more than GK_QSORT_SMALL + 1 elements,
 *     pivot = median of the first, middle and last elements (ordered in
 *     place), ranges kept on an explicit stack, the smaller side sorted
 *     next and ranges of at most GK_QSORT_SMALL + 1 elements left alone;
 *   - then one insertion-sort pass over the whole array, after moving the
 *     smallest of its first GK_QSORT_SMALL + 1 elements to the front as a
 *     sentinel.
 *
 * Usage: GK_MKQSORT(TYPE, BASE, NELT, LT) sorts the NELT elements of type
 * TYPE at BASE into ascending order of LT, where LT(a, b) gets two TYPE
 * pointers and is nonzero when *a sorts before *b (LT is the only
 * comparison made). BASE and NELT are evaluated once.
 */

#ifndef _GK_MKSORT_H_
#define _GK_MKSORT_H_

/* Ranges of at most GK_QSORT_SMALL + 1 elements are left to the final
   insertion sort (the original's threshold). */
#define GK_QSORT_SMALL 8

/* One stack entry per pending range; the smaller side is always handled
   first, so the stack never holds more than log2(n) entries. */
#define GK_QSORT_STACK (8 * sizeof(size_t))

#define GK_QSORT_SWAP(p, q, tmp) \
  do { (tmp) = *(p); *(p) = *(q); *(q) = (tmp); } while (0)

#define GK_MKQSORT(GKQS_T, GKQS_BASE, GKQS_N, GKQS_LT)                        \
do {                                                                          \
  GKQS_T *const gkqs_a = (GKQS_BASE);                                         \
  const size_t gkqs_n = (GKQS_N);                                             \
  GKQS_T gkqs_tmp;                                                            \
                                                                              \
  if (gkqs_n == 0)                                                            \
    break;                                                                    \
                                                                              \
  /* Partition phase. */                                                     \
  if (gkqs_n > GK_QSORT_SMALL) {                                              \
    struct { GKQS_T *first, *last; } gkqs_stack[GK_QSORT_STACK];              \
    size_t gkqs_depth = 0;                                                    \
    GKQS_T *gkqs_first = gkqs_a;                                              \
    GKQS_T *gkqs_last = gkqs_a + gkqs_n - 1;                                  \
                                                                              \
    for (;;) {                                                                \
      GKQS_T *gkqs_piv = gkqs_first + ((gkqs_last - gkqs_first) >> 1);        \
      GKQS_T *gkqs_l, *gkqs_r;                                                \
                                                                              \
      /* first <= pivot <= last, by three compare-and-swaps */                \
      if (GKQS_LT(gkqs_piv, gkqs_first))                                      \
        GK_QSORT_SWAP(gkqs_piv, gkqs_first, gkqs_tmp);                        \
      if (GKQS_LT(gkqs_last, gkqs_piv)) {                                     \
        GK_QSORT_SWAP(gkqs_piv, gkqs_last, gkqs_tmp);                         \
        if (GKQS_LT(gkqs_piv, gkqs_first))                                    \
          GK_QSORT_SWAP(gkqs_piv, gkqs_first, gkqs_tmp);                      \
      }                                                                       \
                                                                              \
      /* Scan inwards from both ends; the pivot element moves with swaps. */  \
      gkqs_l = gkqs_first + 1;                                                \
      gkqs_r = gkqs_last - 1;                                                 \
      while (gkqs_l <= gkqs_r) {                                              \
        while (GKQS_LT(gkqs_l, gkqs_piv))                                     \
          gkqs_l++;                                                           \
        while (GKQS_LT(gkqs_piv, gkqs_r))                                     \
          gkqs_r--;                                                           \
        if (gkqs_l == gkqs_r) {                                               \
          gkqs_l++;                                                           \
          gkqs_r--;                                                           \
          break;                                                              \
        }                                                                     \
        if (gkqs_l > gkqs_r)                                                  \
          break;                                                              \
        GK_QSORT_SWAP(gkqs_l, gkqs_r, gkqs_tmp);                              \
        if (gkqs_piv == gkqs_l)                                               \
          gkqs_piv = gkqs_r;                                                  \
        else if (gkqs_piv == gkqs_r)                                          \
          gkqs_piv = gkqs_l;                                                  \
        gkqs_l++;                                                             \
        gkqs_r--;                                                             \
      }                                                                       \
                                                                              \
      /* [first, r] and [l, last] remain; small ones are left as they are. */ \
      {                                                                       \
        const int gkqs_lo_big = (gkqs_r - gkqs_first) > (ptrdiff_t)GK_QSORT_SMALL; \
        const int gkqs_hi_big = (gkqs_last - gkqs_l) > (ptrdiff_t)GK_QSORT_SMALL;  \
        if (gkqs_lo_big && gkqs_hi_big) {                                     \
          if (gkqs_r - gkqs_first > gkqs_last - gkqs_l) {                     \
            gkqs_stack[gkqs_depth].first = gkqs_first;                        \
            gkqs_stack[gkqs_depth].last = gkqs_r;                             \
            gkqs_depth++;                                                     \
            gkqs_first = gkqs_l;                                              \
          }                                                                   \
          else {                                                              \
            gkqs_stack[gkqs_depth].first = gkqs_l;                            \
            gkqs_stack[gkqs_depth].last = gkqs_last;                          \
            gkqs_depth++;                                                     \
            gkqs_last = gkqs_r;                                               \
          }                                                                   \
        }                                                                     \
        else if (gkqs_lo_big) {                                               \
          gkqs_last = gkqs_r;                                                 \
        }                                                                     \
        else if (gkqs_hi_big) {                                               \
          gkqs_first = gkqs_l;                                                \
        }                                                                     \
        else {                                                                \
          if (gkqs_depth == 0)                                                \
            break;                                                            \
          gkqs_depth--;                                                       \
          gkqs_first = gkqs_stack[gkqs_depth].first;                          \
          gkqs_last = gkqs_stack[gkqs_depth].last;                            \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }                                                                           \
                                                                              \
  /* Insertion-sort phase, with the minimum of the first block in front. */   \
  {                                                                           \
    GKQS_T *const gkqs_end = gkqs_a + gkqs_n - 1;                             \
    GKQS_T *const gkqs_scan_end =                                             \
        gkqs_n - 1 < GK_QSORT_SMALL ? gkqs_end : gkqs_a + GK_QSORT_SMALL;     \
    GKQS_T *gkqs_min = gkqs_a, *gkqs_i, *gkqs_j;                              \
                                                                              \
    for (gkqs_i = gkqs_a + 1; gkqs_i <= gkqs_scan_end; gkqs_i++)              \
      if (GKQS_LT(gkqs_i, gkqs_min))                                          \
        gkqs_min = gkqs_i;                                                    \
    if (gkqs_min != gkqs_a)                                                   \
      GK_QSORT_SWAP(gkqs_min, gkqs_a, gkqs_tmp);                              \
                                                                              \
    /* a[0] is now a lower bound for everything: no left-end check needed. */ \
    for (gkqs_i = gkqs_a + 2; gkqs_i <= gkqs_end; gkqs_i++) {                 \
      gkqs_j = gkqs_i - 1;                                                    \
      while (GKQS_LT(gkqs_i, gkqs_j))                                         \
        gkqs_j--;                                                             \
      gkqs_j++;                                                               \
      if (gkqs_j != gkqs_i) {                                                 \
        GKQS_T *gkqs_k;                                                       \
        gkqs_tmp = *gkqs_i;                                                   \
        for (gkqs_k = gkqs_i; gkqs_k > gkqs_j; gkqs_k--)                      \
          *gkqs_k = *(gkqs_k - 1);                                            \
        *gkqs_j = gkqs_tmp;                                                   \
      }                                                                       \
    }                                                                         \
  }                                                                           \
} while (0)

#endif /* _GK_MKSORT_H_ */
