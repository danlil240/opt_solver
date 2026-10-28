#include "smf/supernode.hpp"
#include <algorithm>
#include <cstdint>

namespace smf
{

    // Relaxed amalgamation bound: a merge is taken when it adds at most
    // kAmalgamateFreeZeros stored zeros, or when the merged node stays at
    // most kAmalgamateMaxZeroShare zeros. Tuned on optim20 (one-shot solves,
    // where merging tiny nodes saves per-node overhead) against the IPOPT
    // KKT sequences (factor -14%, solve -19% vs unbounded merging).
    static constexpr LongInt kAmalgamateFreeZeros = 16;
    static constexpr double kAmalgamateMaxZeroShare = 0.3;

    // Column counts by Gilbert, Ng & Peyton (1994), as CSparse's cs_counts:
    // each row subtree contributes only through its skeleton (leaves) and
    // least common ancestors, found with path-compressed union-find, so the
    // cost is O(nnz(A) alpha(n)) instead of one step per entry of L (the
    // row-subtree walk this replaces cost 20% of analysis on bmwcra_1).
    // Needs parent[j] > j, true for the etree of a permuted matrix.
    // for_each_below(j, f) calls f(i) for every row i > j of column j of the
    // strict lower triangle, in any order (the result does not depend on
    // it). work: 4n ints; delta (the counts) n ints.
    template <class ForEachBelow>
    static void gnp_column_counts(Int n, const Int *parent, const Int *post, Int *delta,
                                  Int *work, ForEachBelow &&for_each_below)
    {
        const auto sn = static_cast<std::size_t>(n);
        std::fill(work, work + 4 * sn, -1);
        Int *ancestor = work;
        Int *maxfirst = ancestor + sn;
        Int *prevleaf = maxfirst + sn;
        Int *first = prevleaf + sn;
        for (Int k = 0; k < n; ++k)
        {
            Int j = post[k];
            delta[j] = (first[j] == -1) ? 1 : 0; // leaf
            for (; j != -1 && first[j] == -1; j = parent[j])
                first[j] = k;
        }
        for (Int i = 0; i < n; ++i)
            ancestor[i] = i;
        for (Int k = 0; k < n; ++k)
        {
            const Int j = post[k];
            if (parent[j] != -1)
                --delta[parent[j]];
            for_each_below(j, [&](Int i) {
                if (first[j] <= maxfirst[i])
                    return; // j is not a leaf of row subtree i
                maxfirst[i] = first[j];
                const Int jprev = prevleaf[i];
                prevleaf[i] = j;
                ++delta[j]; // A(i, j) is in the skeleton
                if (jprev == -1)
                    return; // first leaf of row subtree i
                Int q = jprev;
                while (q != ancestor[q])
                    q = ancestor[q];
                for (Int s = jprev; s != q;)
                {
                    const Int sparent = ancestor[s];
                    ancestor[s] = q;
                    s = sparent;
                }
                --delta[q]; // q = lca(jprev, j) was counted twice
            });
            if (parent[j] != -1)
                ancestor[j] = parent[j];
        }
        for (Int j = 0; j < n; ++j)
            if (parent[j] != -1)
                delta[parent[j]] += delta[j];
    }

    std::vector<Int> filled_column_counts_from_rows(Int n, const Int *row_ptr,
                                                    const Int *row_cols,
                                                    const EliminationTree &etree)
    {
        const auto sn = static_cast<std::size_t>(n);

        // Column form of the strict lower triangle: rows i > j of column j.
        std::vector<Int> cp(sn + 1, 0);
        for (Int i = 0; i < n; ++i)
            for (Int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
                if (row_cols[p] < i)
                    ++cp[static_cast<std::size_t>(row_cols[p]) + 1];
        for (std::size_t j = 0; j < sn; ++j)
            cp[j + 1] += cp[j];
        std::vector<Int> ci(static_cast<std::size_t>(cp[sn]));
        {
            std::vector<Int> next(cp.begin(), cp.end() - 1);
            for (Int i = 0; i < n; ++i)
                for (Int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
                    if (row_cols[p] < i)
                        ci[static_cast<std::size_t>(next[static_cast<std::size_t>(row_cols[p])]++)] = i;
        }

        std::vector<Int> w(4 * sn);
        std::vector<Int> delta(sn);
        gnp_column_counts(n, etree.parent.data(), etree.postorder.data(), delta.data(), w.data(),
                          [&](Int j, auto &&f) {
                              for (Int p = cp[static_cast<std::size_t>(j)];
                                   p < cp[static_cast<std::size_t>(j) + 1]; ++p)
                                  f(ci[static_cast<std::size_t>(p)]);
                          });
        return delta;
    }

    void filled_column_counts_from_adjacency(Int n, const Int *xadj, const Int *adj,
                                             const Int *perm, const Int *iperm,
                                             const Int *parent, const Int *postorder,
                                             Int *colcount, Int *work)
    {
        // Column j of P A P^T below the diagonal: the neighbours of old
        // vertex perm[j] numbered above j. No permuted copy is formed.
        gnp_column_counts(n, parent, postorder, colcount, work, [&](Int j, auto &&f) {
            const Int v = perm[j];
            for (Int p = xadj[v]; p < xadj[v + 1]; ++p)
            {
                const Int i = iperm[adj[p]];
                if (i > j)
                    f(i);
            }
        });
    }

    // Compute exact simplicial column counts |L_{*w}| of the Cholesky factor.
    // Characterization (Liu 1990): i in Struct(L_{*w}) iff A(i,k) != 0 for some
    // k whose etree path to i passes through w.  For each row i we therefore
    // walk the path k -> i from every column k containing i, crediting row i
    // to each vertex on the path.  mark[w] == i means row i was already
    // credited to column w; walks for a given i stop at the first credited
    // vertex because the whole suffix up to i is then already credited as
    // well.  Grouping the walks per i (outer loop) is what makes this
    // early-stop dedup valid.  Time O(nnz + union-of-path lengths).
    static std::vector<Int> filled_col_counts(Int n,
                                                const std::vector<Int> &col_ptr,
                                                const std::vector<Int> &row_idx,
                                                const EliminationTree &etree)
    {
        const auto sn = static_cast<std::size_t>(n);

        // Transpose lower CSC -> upper CSR: for each row i, the columns k < i
        // with A(i,k) != 0 (subdiagonal entries only; diagonals skipped).
        std::vector<Int> ucol(sn + 1, 0);
        for (Int k = 0; k < n; ++k)
            for (Int p = col_ptr[static_cast<std::size_t>(k)];
                 p < col_ptr[static_cast<std::size_t>(k) + 1]; ++p)
            {
                const Int i = row_idx[static_cast<std::size_t>(p)];
                if (i > k)
                    ++ucol[static_cast<std::size_t>(i) + 1];
            }
        for (Int i = 0; i < n; ++i)
            ucol[static_cast<std::size_t>(i) + 1] +=
                ucol[static_cast<std::size_t>(i)];
        std::vector<Int> urow(static_cast<std::size_t>(ucol[sn]));
        {
            std::vector<Int> pos(ucol.begin(),
                                   ucol.begin() + static_cast<std::ptrdiff_t>(n));
            for (Int k = 0; k < n; ++k)
                for (Int p = col_ptr[static_cast<std::size_t>(k)];
                     p < col_ptr[static_cast<std::size_t>(k) + 1]; ++p)
                {
                    const Int i = row_idx[static_cast<std::size_t>(p)];
                    if (i > k)
                        urow[static_cast<std::size_t>(
                            pos[static_cast<std::size_t>(i)]++)] = k;
                }
        }

        return filled_column_counts_from_rows(n, ucol.data(), urow.data(), etree);
    }

    std::vector<Supernode> detect_fundamental_supernodes(Int n,
                                                         const std::vector<Int> &col_ptr,
                                                         const std::vector<Int> &row_idx,
                                                         const EliminationTree &etree)
    {
        if (n == 0)
            return {};

        // Count etree children for each node
        std::vector<Int> nchildren(static_cast<std::size_t>(n), 0);
        for (Int j = 0; j < n; ++j)
        {
            const Int p = etree.parent[static_cast<std::size_t>(j)];
            if (p >= 0)
                ++nchildren[static_cast<std::size_t>(p)];
        }

        // Filled column counts |L_{*j}| (structure of the factor, not of A).
        // Using original column counts fragments large ordered separators into
        // singleton supernodes; filled counts recover the true fundamental
        // supernodes (Liu-Ng-Peyton): j continues the supernode of j-1 iff
        //   (1) etree.parent[j-1] == j
        //   (2) j-1 is j's only etree child
        //   (3) |L_{*j-1}| == |L_{*j}| + 1
        std::vector<Int> colcount =
            filled_col_counts(n, col_ptr, row_idx, etree);

        std::vector<Int> snode_id(static_cast<std::size_t>(n), -1);
        std::vector<Int> snode_start; // first column of each supernode
        snode_start.reserve(static_cast<std::size_t>(n));
        for (Int j = 0; j < n; ++j)
        {
            bool new_snode = (j == 0);
            if (!new_snode)
            {
                const Int par_prev = etree.parent[static_cast<std::size_t>(j - 1)];
                new_snode = (par_prev != j) ||
                            (nchildren[static_cast<std::size_t>(j)] != 1) ||
                            (colcount[static_cast<std::size_t>(j - 1)] !=
                             colcount[static_cast<std::size_t>(j)] + 1);
            }
            if (new_snode)
            {
                snode_id[static_cast<std::size_t>(j)] = static_cast<Int>(snode_start.size());
                snode_start.push_back(j);
            }
            else
            {
                snode_id[static_cast<std::size_t>(j)] = static_cast<Int>(snode_start.size()) - 1;
            }
        }

        const Int nsn = static_cast<Int>(snode_start.size());
        std::vector<Supernode> result(static_cast<std::size_t>(nsn));
        for (Int s = 0; s < nsn; ++s)
        {
            result[static_cast<std::size_t>(s)].col_start = snode_start[static_cast<std::size_t>(s)];
            result[static_cast<std::size_t>(s)].col_end =
                (s + 1 < nsn) ? snode_start[static_cast<std::size_t>(s) + 1] : n;
            result[static_cast<std::size_t>(s)].parent = -1;
        }

        // Build supernode tree: parent of SN s = SN containing etree.parent[last col of s]
        for (Int s = 0; s < nsn; ++s)
        {
            const Int last_col = result[static_cast<std::size_t>(s)].col_end - 1;
            const Int par_col = etree.parent[static_cast<std::size_t>(last_col)];
            if (par_col < 0)
                continue;
            const Int par_sn = snode_id[static_cast<std::size_t>(par_col)];
            result[static_cast<std::size_t>(s)].parent = par_sn;
        }

        return result;
    }

    std::vector<Supernode> amalgamate_supernodes(std::vector<Supernode> supernodes, Int nemin)
    {
        if (nemin <= 1)
            return supernodes;

        const Int nsn = static_cast<Int>(supernodes.size());

        // Single bottom-up pass: parents always have larger indices than
        // children (etree property), so cascading merges settle in one sweep.
        // new_parent[c] tracks each node's current effective parent; dead
        // supernodes are skipped through when the final tree is rebuilt.
        std::vector<Int> new_parent(static_cast<std::size_t>(nsn));
        std::vector<char> dead(static_cast<std::size_t>(nsn), 0);
        for (Int s = 0; s < nsn; ++s)
            new_parent[static_cast<std::size_t>(s)] =
                supernodes[static_cast<std::size_t>(s)].parent;

        for (Int s = 0; s < nsn; ++s)
        {
            if (dead[static_cast<std::size_t>(s)])
                continue;
            const Int p = new_parent[static_cast<std::size_t>(s)];
            if (p < 0 || dead[static_cast<std::size_t>(p)])
                continue;

            if (supernodes[static_cast<std::size_t>(s)].width() < nemin &&
                supernodes[static_cast<std::size_t>(p)].width() < nemin &&
                supernodes[static_cast<std::size_t>(s)].col_end ==
                    supernodes[static_cast<std::size_t>(p)].col_start)
            {
                // Merge s into p: absorb s's column range. s's children (all
                // already visited: smaller indices) keep pointing at s; the
                // parent resolution below skips through dead nodes to p.
                supernodes[static_cast<std::size_t>(p)].col_start =
                    supernodes[static_cast<std::size_t>(s)].col_start;
                dead[static_cast<std::size_t>(s)] = 1;
            }
        }

        // Collect survivors (sorted by col_start preserves postorder)
        std::vector<Int> valid;
        valid.reserve(static_cast<std::size_t>(nsn));
        for (Int s = 0; s < nsn; ++s)
            if (!dead[static_cast<std::size_t>(s)])
                valid.push_back(s);

        const Int nr = static_cast<Int>(valid.size());

        // Resolve each node's effective parent through dead supernodes
        std::vector<Int> orig_to_ri(static_cast<std::size_t>(nsn), -1);
        for (Int i = 0; i < nr; ++i)
            orig_to_ri[static_cast<std::size_t>(valid[static_cast<std::size_t>(i)])] = i;

        std::vector<Supernode> result(static_cast<std::size_t>(nr));
        for (Int i = 0; i < nr; ++i)
        {
            const Int s = valid[static_cast<std::size_t>(i)];
            result[static_cast<std::size_t>(i)].col_start = supernodes[static_cast<std::size_t>(s)].col_start;
            result[static_cast<std::size_t>(i)].col_end = supernodes[static_cast<std::size_t>(s)].col_end;
            result[static_cast<std::size_t>(i)].parent = -1;
        }

        for (Int i = 0; i < nr; ++i)
        {
            const Int s = valid[static_cast<std::size_t>(i)];
            Int par = new_parent[static_cast<std::size_t>(s)];
            while (par >= 0 && dead[static_cast<std::size_t>(par)])
                par = new_parent[static_cast<std::size_t>(par)];
            if (par < 0)
                continue;
            const Int ri_par = orig_to_ri[static_cast<std::size_t>(par)];
            result[static_cast<std::size_t>(i)].parent = ri_par;
        }

        return result;
    }

    std::size_t amalgamate_relaxed_work_size(Int n)
    {
        return 16 * static_cast<std::size_t>(n) + 6;
    }

    void amalgamate_relaxed(Int n, const Int *parent, const Int *colcount, Int nemin,
                            std::vector<Int> &order, std::vector<Supernode> &snodes,
                            const std::uint8_t *keep_with_parent, Int *work,
                            Amalgamation rule)
    {
        order.resize(static_cast<std::size_t>(n));
        snodes.clear();
        if (n == 0)
            return;
        const auto sn = static_cast<std::size_t>(n);

        // Every size-n / size-nsn temporary lives in one block (two-phase:
        // nsn is only known after the fundamental pass): the caller's, or
        // an own one. Only the counters below are zeroed.
        std::vector<Int> own;
        if (work == nullptr)
        {
            own.resize(amalgamate_relaxed_work_size(n));
            work = own.data();
        }
        Int *nchildren = work;                 // n
        Int *snode_of = nchildren + sn;        // n
        Int *first = snode_of + sn;            // nsn + 1 <= n + 1
        std::fill(nchildren, nchildren + n, 0);

        // Fundamental supernodes of the postordered numbering.
        for (Int j = 0; j < n; ++j)
            if (parent[static_cast<std::size_t>(j)] >= 0)
                ++nchildren[static_cast<std::size_t>(parent[static_cast<std::size_t>(j)])];
        Int nsn = 0;
        for (Int j = 0; j < n; ++j)
        {
            const bool new_snode =
                (j == 0) || (parent[static_cast<std::size_t>(j) - 1] != j) ||
                (nchildren[static_cast<std::size_t>(j)] != 1) ||
                (colcount[static_cast<std::size_t>(j) - 1] !=
                 colcount[static_cast<std::size_t>(j)] + 1);
            if (new_snode)
                first[nsn++] = j;
            snode_of[static_cast<std::size_t>(j)] = nsn - 1;
        }
        first[nsn] = n;
        const auto nsz = static_cast<std::size_t>(nsn);
        Int *swork = first + sn + 1;           // 11 nsn + 2
        Int *sparent = swork;                  // nsn
        Int *nelim = sparent + nsz;            // nsn
        Int *group = nelim + nsz;              // nsn
        Int *gparent = group + nsz;            // nsn
        Int *ch_ptr = gparent + nsz;           // nsn + 1
        Int *ch = ch_ptr + nsz + 1;            // nsn
        Int *gcol_ptr = ch + nsz;              // nsn + 1
        Int *new_id = gcol_ptr + nsz + 1;      // nsn
        Int *post = new_id + nsz;              // nsn
        Int *stk_node = post + nsz;            // nsn
        Int *stk_ci = stk_node + nsz;          // nsn
        // Per-group fill (LongInt, nsn), 8-byte aligned past stk_ci.
        Int *fill_base = stk_ci + nsz;
        if ((reinterpret_cast<std::uintptr_t>(fill_base) % alignof(LongInt)) != 0)
            ++fill_base;
        LongInt *fill = reinterpret_cast<LongInt *>(fill_base);
        std::fill(ch_ptr, ch_ptr + nsn + 1, 0);
        std::fill(gcol_ptr, gcol_ptr + nsn + 1, 0);
        std::fill(sparent, sparent + nsn, -1);
        for (Int s = 0; s < nsn; ++s)
        {
            const auto si = static_cast<std::size_t>(s);
            nelim[si] = first[si + 1] - first[si];
            const Int pc = parent[static_cast<std::size_t>(first[si + 1]) - 1];
            if (pc >= 0)
                sparent[si] = snode_of[static_cast<std::size_t>(pc)];
        }

        // Merge bottom-up (children have smaller indices). group[s] is the
        // supernode s was merged into; survivors point to themselves.
        //
        // Both must have fewer than nemin columns (MA97's rule), and, as in
        // CHOLMOD's relaxed supernodes, the merged node may carry only a
        // bounded share of explicit zeros. Its rows are its columns plus the
        // rows below the parent's last column (a child's structure is inside
        // the parent's), so for k columns and q such rows the front stores
        // k (k + q) values (column-major, the pivot block's upper triangle
        // included), of which the column counts are true fill. Without the
        // bound, KKT trees of tiny supernodes doubled the factor with zeros
        // (pde_control_120: 2.46M stored values vs 1.27M), which every solve
        // then streams through.
        for (Int s = 0; s < nsn; ++s)
            group[static_cast<std::size_t>(s)] = s;
        // Forced merges (keep_with_parent): a flagged column whose parent
        // lies in another fundamental supernode is that supernode's last
        // column (inside one, parent[j] == j + 1), so the supernode's own
        // parent is the column's parent's supernode.
        std::vector<std::uint8_t> forced;
        if (keep_with_parent != nullptr)
        {
            forced.assign(nsz, 0);
            for (Int j = 0; j < n; ++j)
            {
                const Int pj = parent[static_cast<std::size_t>(j)];
                const Int sj = snode_of[static_cast<std::size_t>(j)];
                if (keep_with_parent[static_cast<std::size_t>(j)] != 0 && pj >= 0 &&
                    snode_of[static_cast<std::size_t>(pj)] != sj)
                    forced[static_cast<std::size_t>(sj)] = 1;
            }
        }
        if (nemin > 1 || !forced.empty())
        {
            std::fill(fill, fill + nsn, LongInt{0});
            for (Int j = 0; j < n; ++j)
                fill[static_cast<std::size_t>(snode_of[static_cast<std::size_t>(j)])] +=
                    colcount[static_cast<std::size_t>(j)];
            for (Int s = 0; s < nsn; ++s)
            {
                const auto si = static_cast<std::size_t>(s);
                const Int p = sparent[si];
                if (p < 0)
                    continue;
                const auto pi = static_cast<std::size_t>(p);
                if (forced.empty() || forced[si] == 0)
                {
                    const auto stored = [&](std::size_t g, LongInt cols) {
                        const LongInt q = colcount[static_cast<std::size_t>(first[g + 1]) - 1] - 1;
                        return cols * (cols + q);
                    };
                    const LongInt k = nelim[si] + nelim[pi];
                    const LongInt entries = stored(pi, k);
                    const LongInt zeros = entries - fill[si] - fill[pi];
                    const LongInt added = entries - stored(si, nelim[si]) - stored(pi, nelim[pi]);
                    bool take = false;
                    if (nelim[si] < nemin && nelim[pi] < nemin)
                        take = !(added > kAmalgamateFreeZeros &&
                                 static_cast<double>(zeros) >
                                     kAmalgamateMaxZeroShare * static_cast<double>(entries));
                    if (!take && rule == Amalgamation::SmfPlusCholmod && nemin > 1)
                    {
                        // CHOLMOD's default relaxation (nrelax 4/16/48, zrelax
                        // 0.8/0.1/0.05), independent of the nemin gate (nemin 1 still means
                        // no amalgamation).
                        const double z = static_cast<double>(zeros) / static_cast<double>(entries);
                        take = k <= 4 || added <= 0 || (k <= 16 && z < 0.8) ||
                               (k <= 48 && z < 0.1) || z < 0.05;
                    }
                    if (!take)
                        continue;
                }
                group[si] = p;
                nelim[pi] += nelim[si];
                fill[pi] += fill[si];
            }
        }
        // Resolve to survivors: a target always has a larger index than the
        // merged node, so a descending sweep sees every target resolved.
        for (Int s = nsn - 1; s >= 0; --s)
        {
            const auto si = static_cast<std::size_t>(s);
            group[si] = group[static_cast<std::size_t>(group[si])];
        }

        // Group tree and its children (in increasing survivor index).
        std::fill(gparent, gparent + nsn, -1);
        for (Int g = 0; g < nsn; ++g)
        {
            if (group[static_cast<std::size_t>(g)] != g)
                continue;
            const Int p = sparent[static_cast<std::size_t>(g)];
            if (p >= 0)
            {
                gparent[static_cast<std::size_t>(g)] = group[static_cast<std::size_t>(p)];
                ++ch_ptr[static_cast<std::size_t>(gparent[static_cast<std::size_t>(g)]) + 1];
            }
        }
        for (Int g = 0; g < nsn; ++g)
            ch_ptr[static_cast<std::size_t>(g) + 1] += ch_ptr[static_cast<std::size_t>(g)];
        {
            Int *head = new_id; // free until the DFS below
            std::copy(ch_ptr, ch_ptr + nsn, head);
            for (Int g = 0; g < nsn; ++g)
                if (group[static_cast<std::size_t>(g)] == g && gparent[static_cast<std::size_t>(g)] >= 0)
                    ch[head[gparent[static_cast<std::size_t>(g)]]++] = g;
        }

        // Columns of each group in increasing input index.
        for (Int s = 0; s < nsn; ++s)
            gcol_ptr[static_cast<std::size_t>(group[static_cast<std::size_t>(s)]) + 1] +=
                first[static_cast<std::size_t>(s) + 1] - first[static_cast<std::size_t>(s)];
        for (Int g = 0; g < nsn; ++g)
            gcol_ptr[static_cast<std::size_t>(g) + 1] += gcol_ptr[static_cast<std::size_t>(g)];
        Int *gcols = nchildren; // nchildren is dead; reuse its n slots
        {
            Int *head = new_id; // free until the DFS below
            std::copy(gcol_ptr, gcol_ptr + nsn, head);
            for (Int j = 0; j < n; ++j)
            {
                const Int g = group[static_cast<std::size_t>(snode_of[static_cast<std::size_t>(j)])];
                gcols[head[g]++] = j;
            }
        }

        // Postorder DFS of the group tree; each group's columns are emitted
        // contiguously, which keeps the order topological.
        std::fill(new_id, new_id + nsn, -1);
        Int ng = 0;
        {
            Int top = 0;
            for (Int g = nsn - 1; g >= 0; --g)
                if (group[static_cast<std::size_t>(g)] == g && gparent[static_cast<std::size_t>(g)] < 0)
                {
                    stk_node[top] = g;
                    stk_ci[top] = 0;
                    ++top;
                }
            while (top > 0)
            {
                const Int g = stk_node[top - 1];
                const Int c0 = ch_ptr[g];
                const Int c1 = ch_ptr[g + 1];
                const Int ci = stk_ci[top - 1];
                if (c0 + ci < c1)
                {
                    stk_ci[top - 1] = ci + 1;
                    stk_node[top] = ch[c0 + ci];
                    stk_ci[top] = 0;
                    ++top;
                }
                else
                {
                    new_id[g] = ng;
                    post[ng++] = g;
                    --top;
                }
            }
        }

        snodes.resize(static_cast<std::size_t>(ng));
        Int pos = 0;
        for (Int i = 0; i < ng; ++i)
        {
            const Int g = post[static_cast<std::size_t>(i)];
            Supernode &out = snodes[static_cast<std::size_t>(i)];
            out.col_start = pos;
            for (Int t = gcol_ptr[static_cast<std::size_t>(g)]; t < gcol_ptr[static_cast<std::size_t>(g) + 1]; ++t)
                order[static_cast<std::size_t>(pos++)] = gcols[static_cast<std::size_t>(t)];
            out.col_end = pos;
            const Int gp = gparent[static_cast<std::size_t>(g)];
            out.parent = (gp >= 0) ? new_id[static_cast<std::size_t>(gp)] : -1;
        }
    }

    void build_children_csr(std::span<const Supernode> snodes,
                            std::vector<Int> &child_ptr, std::vector<Int> &child_idx)
    {
        const std::size_t nsn = snodes.size();
        child_ptr.assign(nsn + 1, 0);
        // Pass 1: count each parent's children.
        for (std::size_t s = 0; s < nsn; ++s)
        {
            const Int par = snodes[s].parent;
            if (par >= 0)
                ++child_ptr[static_cast<std::size_t>(par) + 1];
        }
        for (std::size_t s = 0; s < nsn; ++s)
            child_ptr[s + 1] += child_ptr[s];
        child_idx.resize(static_cast<std::size_t>(child_ptr[nsn]));
        // Pass 2: fill each parent's slots from the back while visiting the
        // children in increasing index, which leaves them in descending order.
        // child_ptr[par + 1] serves as the moving end and is restored after.
        for (std::size_t s = 0; s < nsn; ++s)
        {
            const Int par = snodes[s].parent;
            if (par >= 0)
            {
                const Int slot = --child_ptr[static_cast<std::size_t>(par) + 1];
                child_idx[static_cast<std::size_t>(slot)] = static_cast<Int>(s);
            }
        }
        // Each end pointer child_ptr[p + 1] was decremented down to p's
        // start, i.e. now holds the original child_ptr[p]: shift down by one.
        const Int total = static_cast<Int>(child_idx.size());
        for (std::size_t s = 0; s < nsn; ++s)
            child_ptr[s] = child_ptr[s + 1];
        child_ptr[nsn] = total;
    }

} // namespace smf
