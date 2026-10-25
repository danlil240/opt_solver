#include "smf/supernode.hpp"
#include <algorithm>

namespace smf
{

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
        std::vector<Int> count(sn, 1); // +1 for the diagonal entry
        std::vector<Int> mark(sn, -1);

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

        // For each row i: every k in urow[i] contributes i to all vertices on
        // the etree path k -> i (excluding i itself, the diagonal slot).
        for (Int i = 0; i < n; ++i)
        {
            for (Int p = ucol[static_cast<std::size_t>(i)];
                 p < ucol[static_cast<std::size_t>(i) + 1]; ++p)
            {
                Int w = urow[static_cast<std::size_t>(p)];
                while (w >= 0 && w != i &&
                       mark[static_cast<std::size_t>(w)] != i)
                {
                    mark[static_cast<std::size_t>(w)] = i;
                    ++count[static_cast<std::size_t>(w)];
                    w = etree.parent[static_cast<std::size_t>(w)];
                }
            }
        }
        return count;
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
            result[static_cast<std::size_t>(par_sn)].children.push_back(s);
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
                // Merge s into p: absorb s's column range; s's children become
                // children of p (their new_parent is re-pointed below).
                supernodes[static_cast<std::size_t>(p)].col_start =
                    supernodes[static_cast<std::size_t>(s)].col_start;
                dead[static_cast<std::size_t>(s)] = 1;
                for (Int c : supernodes[static_cast<std::size_t>(s)].children)
                    new_parent[static_cast<std::size_t>(c)] = p;
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
            if (ri_par >= 0)
                result[static_cast<std::size_t>(ri_par)].children.push_back(i);
        }

        return result;
    }

} // namespace smf
