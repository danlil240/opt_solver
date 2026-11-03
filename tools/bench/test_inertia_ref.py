"""Tests for tools/bench/inertia_ref.py helpers (no bench_replay needed)."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import inertia_ref  # noqa: E402


def lower_coo(A):
    r, c = np.nonzero(np.tril(A))
    return r.astype(np.int64), c.astype(np.int64), A[r, c].astype(float)


class TestInertiaRef(unittest.TestCase):
    def test_equilibrate_keeps_inertia_and_scales_rows(self):
        # KKT-like, badly scaled: barrier-sized entries next to O(1) ones.
        A = np.array([[1e12, 0, 1.0, 0],
                      [0, 2.0, 1.0, 0],
                      [1.0, 1.0, 0, 3e-6],
                      [0, 0, 3e-6, -1e-8]])
        r, c, v = lower_coo(A)
        e = inertia_ref.equilibrate(4, r, c, v)
        B = np.zeros((4, 4))
        B[r, c] = e
        B[c, r] = e
        self.assertTrue(np.allclose(np.max(np.abs(B), axis=1), 1.0, atol=1e-2))
        sa = np.sign(np.linalg.eigvalsh(A))
        sb = np.sign(np.linalg.eigvalsh(B))
        self.assertEqual(sorted(sa), sorted(sb))

    def test_dense_counts_zero_cluster(self):
        A = np.diag([3.0, 1.0, 1e-14, -1e-14, -2.0])
        r, c, v = lower_coo(A)
        got = inertia_ref.dense_counts(5, r, c, v, 1e-10)
        self.assertEqual((got["pos"], got["neg"], got["zero"]), (2, 1, 2))

    def test_shifted_mtx_adds_missing_diagonal(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        r, c, v = lower_coo(A)
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "s.mtx"
            inertia_ref.write_shifted_mtx(f, 2, r, c, v, 0.5)
            n, r2, c2, v2 = inertia_ref.read_mtx_lower(f)
        B = np.zeros((2, 2))
        B[r2, c2] = v2
        B[c2, r2] = v2
        self.assertEqual(n, 2)
        self.assertTrue(np.allclose(B, A + 0.5 * np.eye(2)))


if __name__ == "__main__":
    unittest.main()
