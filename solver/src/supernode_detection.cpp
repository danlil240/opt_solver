#include "smf/supernode.hpp"
#include <algorithm>

namespace smf
{

    std::vector<Supernode> detect_fundamental_supernodes(Int n,
                                                         const std::vector<Int> &col_ptr,
                                                         const std::vector<Int> & /*row_idx*/,
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

        // col_count[j] = number of structural entries in column j (includes diagonal)
        auto col_count = [&](Int j) -> Int
        { return col_ptr[static_cast<std::size_t>(j) + 1] - col_ptr[static_cast<std::size_t>(j)]; };

        // Assign each column to a supernode.
        // Column j continues the previous supernode iff ALL hold:
        //   (1) etree.parent[j-1] == j   (j is the direct parent of j-1)
        //   (2) nchildren[j] == 1         (j-1 is j's only etree child)
        //   (3) col_count[j-1] == col_count[j] + 1
        std::vector<Int> snode_id(static_cast<std::size_t>(n), -1);
        std::vector<Int> snode_start; // first column of each supernode
        for (Int j = 0; j < n; ++j)
        {
            bool new_snode = (j == 0);
            if (!new_snode)
            {
                const Int par_prev = etree.parent[static_cast<std::size_t>(j - 1)];
                new_snode = (par_prev != j) || (nchildren[static_cast<std::size_t>(j)] != 1) ||
                            (col_count(j - 1) != col_count(j) + 1);
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

        // Iteratively merge small child supernodes into small parents.
        // A supernode is invalid (merged away) when col_start == -1.
        // Because parents always have higher column indices than children (etree
        // property), processing 0..nsn-1 in order lets cascading merges happen
        // within a single pass.  The outer loop repeats until stable.
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (Int s = 0; s < nsn; ++s)
            {
                if (supernodes[static_cast<std::size_t>(s)].col_start < 0)
                    continue;
                const Int p = supernodes[static_cast<std::size_t>(s)].parent;
                if (p < 0)
                    continue;
                if (supernodes[static_cast<std::size_t>(p)].col_start < 0)
                    continue;

                if (supernodes[static_cast<std::size_t>(s)].width() < nemin &&
                    supernodes[static_cast<std::size_t>(p)].width() < nemin &&
                    supernodes[static_cast<std::size_t>(s)].col_end ==
                        supernodes[static_cast<std::size_t>(p)].col_start)
                {
                    // Merge s into p: p absorbs s's column range, replaces child s
                    // with s's children, and keeps any other existing children.
                    supernodes[static_cast<std::size_t>(p)].col_start =
                        supernodes[static_cast<std::size_t>(s)].col_start;

                    std::vector<Int> merged_children;
                    merged_children.reserve(supernodes[static_cast<std::size_t>(p)].children.size() +
                                            supernodes[static_cast<std::size_t>(s)].children.size());

                    for (Int c : supernodes[static_cast<std::size_t>(p)].children)
                        if (c != s)
                            merged_children.push_back(c);
                    for (Int c : supernodes[static_cast<std::size_t>(s)].children)
                        merged_children.push_back(c);

                    supernodes[static_cast<std::size_t>(p)].children = std::move(merged_children);

                    for (Int c : supernodes[static_cast<std::size_t>(p)].children)
                        supernodes[static_cast<std::size_t>(c)].parent = p;

                    // Invalidate s
                    supernodes[static_cast<std::size_t>(s)].col_start = -1;
                    supernodes[static_cast<std::size_t>(s)].col_end = -1;
                    changed = true;
                }
            }
        }

        // Collect surviving supernodes and sort by col_start (preserves postorder)
        std::vector<Int> valid;
        valid.reserve(static_cast<std::size_t>(nsn));
        for (Int s = 0; s < nsn; ++s)
            if (supernodes[static_cast<std::size_t>(s)].col_start >= 0)
                valid.push_back(s);

        std::sort(valid.begin(), valid.end(),
                  [&](Int a, Int b)
                  {
                      return supernodes[static_cast<std::size_t>(a)].col_start <
                             supernodes[static_cast<std::size_t>(b)].col_start;
                  });

        const Int nr = static_cast<Int>(valid.size());

        // Map original SN index → result index
        std::vector<Int> orig_to_ri(static_cast<std::size_t>(nsn), -1);
        for (Int i = 0; i < nr; ++i)
            orig_to_ri[static_cast<std::size_t>(valid[static_cast<std::size_t>(i)])] = i;

        // Build result supernodes
        std::vector<Supernode> result(static_cast<std::size_t>(nr));
        for (Int i = 0; i < nr; ++i)
        {
            const Int s = valid[static_cast<std::size_t>(i)];
            result[static_cast<std::size_t>(i)].col_start = supernodes[static_cast<std::size_t>(s)].col_start;
            result[static_cast<std::size_t>(i)].col_end = supernodes[static_cast<std::size_t>(s)].col_end;
            result[static_cast<std::size_t>(i)].parent = -1;
        }

        // Rebuild parent/children links using the (updated) parent pointers
        for (Int i = 0; i < nr; ++i)
        {
            const Int s = valid[static_cast<std::size_t>(i)];
            const Int orig_par = supernodes[static_cast<std::size_t>(s)].parent;
            if (orig_par < 0)
                continue;
            const Int ri_par = orig_to_ri[static_cast<std::size_t>(orig_par)];
            result[static_cast<std::size_t>(i)].parent = ri_par;
            if (ri_par >= 0)
                result[static_cast<std::size_t>(ri_par)].children.push_back(i);
        }

        return result;
    }

} // namespace smf
