// matrix_market.cpp — Matrix Market (.mtx) file reader implementation

#include "smf/matrix_market.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace smf {

namespace {

// ---------------------------------------------------------------------------
// Internal COO entry
// ---------------------------------------------------------------------------
struct CooEntry {
    Int    row;
    Int    col;
    double val;
};

// ---------------------------------------------------------------------------
// Convert COO (assumed row >= col, sorted or unsorted) to lower-CSC.
// Duplicate (row,col) are summed.
// ---------------------------------------------------------------------------
static CscLower coo_to_lower_csc(int N, std::vector<CooEntry> &coo) {
    // Sort by (col, row)
    std::sort(coo.begin(), coo.end(), [](const CooEntry &a, const CooEntry &b) {
        if (a.col != b.col) return a.col < b.col;
        return a.row < b.row;
    });

    // Merge duplicates
    std::vector<CooEntry> merged;
    merged.reserve(coo.size());
    for (auto &e : coo) {
        if (!merged.empty() &&
            merged.back().row == e.row &&
            merged.back().col == e.col)
        {
            merged.back().val += e.val;
        } else {
            merged.push_back(e);
        }
    }

    CscLower A;
    A.n = static_cast<Int>(N);
    A.col_ptr.assign(static_cast<std::size_t>(N) + 1, 0);

    for (auto &e : merged) {
        A.col_ptr[static_cast<std::size_t>(e.col) + 1]++;
    }
    for (int j = 0; j < N; ++j) {
        A.col_ptr[static_cast<std::size_t>(j) + 1] +=
            A.col_ptr[static_cast<std::size_t>(j)];
    }

    Int nnz_val = A.col_ptr[static_cast<std::size_t>(N)];
    A.row_idx.resize(static_cast<std::size_t>(nnz_val));
    A.values.resize(static_cast<std::size_t>(nnz_val));

    std::vector<Int> pos(static_cast<std::size_t>(N), 0);
    for (auto &e : merged) {
        Int base = A.col_ptr[static_cast<std::size_t>(e.col)];
        Int off  = pos[static_cast<std::size_t>(e.col)];
        A.row_idx[static_cast<std::size_t>(base + off)] = e.row;
        A.values [static_cast<std::size_t>(base + off)] = e.val;
        ++pos[static_cast<std::size_t>(e.col)];
    }
    return A;
}

// ---------------------------------------------------------------------------
// Lowercase a string in-place
// ---------------------------------------------------------------------------
static void to_lower(std::string &s) {
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
MatrixMarketResult read_matrix_market(const char *path) {
    MatrixMarketResult result;

    if (!path || path[0] == '\0') {
        result.error_message = "empty file path";
        return result;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        result.error_message = std::string("cannot open file: ") + path;
        return result;
    }

    bool header_parsed  = false;
    bool size_parsed    = false;
    int  rows           = 0;
    int  cols           = 0;
    int  nnz_file       = 0;
    bool is_symmetric   = false;
    bool is_integer     = false;
    bool is_pattern     = false;

    std::vector<CooEntry> coo;

    std::string line;
    int line_no = 0;

    while (std::getline(ifs, line)) {
        ++line_no;
        // Strip trailing CR (Windows line endings)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // ---- Header line (must be first non-empty line) ----
        if (!header_parsed) {
            if (line.empty()) continue;

            std::string lower = line;
            to_lower(lower);

            // Must start with %%matrixmarket
            if (lower.rfind("%%matrixmarket", 0) != 0) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": not a Matrix Market file (missing %%MatrixMarket banner)";
                return result;
            }

            // Must be "matrix" object
            if (lower.find("matrix") == std::string::npos) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": unsupported object type (expected 'matrix')";
                return result;
            }

            // Format: must be "coordinate"
            if (lower.find("coordinate") == std::string::npos) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": only 'coordinate' format supported (not 'array')";
                return result;
            }

            // Field type
            if (lower.find("pattern") != std::string::npos) {
                is_pattern = true;
            } else if (lower.find("integer") != std::string::npos) {
                is_integer = true;
            } else if (lower.find("real") == std::string::npos &&
                       lower.find("double") == std::string::npos) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": unsupported field type (expected 'real', 'integer', or 'pattern')";
                return result;
            }

            // Symmetry
            if (lower.find("symmetric") != std::string::npos ||
                lower.find("hermitian") != std::string::npos)
            {
                is_symmetric = true;
                result.is_symmetric = true;
            } else {
                result.is_general = true;
            }

            header_parsed = true;
            continue;
        }

        // ---- Comment lines ----
        if (!line.empty() && line[0] == '%') continue;
        if (line.empty()) continue;

        // ---- Size line ----
        if (!size_parsed) {
            std::istringstream ss(line);
            int nrows = 0, ncols = 0, nz = 0;
            if (!(ss >> nrows >> ncols >> nz)) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": failed to parse size line (expected: rows cols nnz)";
                return result;
            }
            if (nrows <= 0 || ncols <= 0 || nz < 0) {
                result.error_message =
                    std::string("line ") + std::to_string(line_no) +
                    ": invalid size: rows=" + std::to_string(nrows) +
                    " cols=" + std::to_string(ncols) +
                    " nnz=" + std::to_string(nz);
                return result;
            }
            rows     = nrows;
            cols     = ncols;
            nnz_file = nz;
            result.rows         = nrows;
            result.cols         = ncols;
            result.nnz_declared = nz;
            coo.reserve(static_cast<std::size_t>(nz));
            size_parsed = true;
            continue;
        }

        // ---- Data lines ----
        std::istringstream ss(line);
        int r = 0, c = 0;
        double v = 0.0;

        if (is_pattern) {
            if (!(ss >> r >> c)) continue;  // skip malformed
            v = 1.0;
        } else if (is_integer) {
            long iv = 0;
            if (!(ss >> r >> c >> iv)) continue;
            v = static_cast<double>(iv);
        } else {
            if (!(ss >> r >> c >> v)) continue;
        }

        // Convert 1-based → 0-based
        r -= 1;
        c -= 1;

        // Discard out-of-range
        if (r < 0 || c < 0 || r >= rows || c >= cols) continue;

        if (is_symmetric) {
            // For symmetric: store lower-triangle entry (row >= col)
            if (r >= c) {
                coo.push_back({static_cast<Int>(r), static_cast<Int>(c), v});
            } else {
                // Mirror: swap to lower triangle
                coo.push_back({static_cast<Int>(c), static_cast<Int>(r), v});
            }
        } else {
            // General: keep only lower-triangle entries
            if (r >= c) {
                coo.push_back({static_cast<Int>(r), static_cast<Int>(c), v});
            }
            // Upper-triangle entries silently discarded
        }
    }

    if (!header_parsed) {
        result.error_message = "file is empty or has no %%MatrixMarket header";
        return result;
    }
    if (!size_parsed) {
        result.error_message = "no size line found after header";
        return result;
    }
    if (rows != cols) {
        result.error_message =
            std::string("matrix is not square: ") +
            std::to_string(rows) + " x " + std::to_string(cols);
        return result;
    }

    (void)nnz_file; // trust actual data count rather than declared count

    result.matrix = coo_to_lower_csc(rows, coo);
    result.ok     = true;
    return result;
}

} // namespace smf
