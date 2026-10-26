import unittest

from ipm_gates import identical_iterations, iteration_lines

HEADER = "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls"
ITERS = [
    "   0  9.0000000e+00 6.37e+01 1.56e-04  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0",
    "   1  1.0003241e+01 5.25e-04 2.71e+03  -1.0 4.19e+00    -  8.99e-01 1.00e+00f  1",
    "   2r 8.7777148e+00 5.03e-04 2.60e+03  -1.0 2.84e+01   0.0 2.61e-01 4.31e-02h  4",
]


def log(iters, t):
    return "\n".join(
        ["This is Ipopt version 3.14.20", HEADER] + iters +
        ["EXIT: Optimal Solution Found.",
         f"[smf_ma97] factor 1: corr=0 delays=3 static=0 pos=5 neg=3 zero=0 L=1000 t={t}",
         f"[smf_ma97] factor 2: corr=1 delays=3 static=0 pos=5 neg=3 zero=0 L=1000 t={t * 2}",
         f"[smf_ma97] factors=2 ({t * 3:.3f} s) solves=3 rhs=3 (0.010 s, 3.3 us/solve)"])


class TestIdenticalIterations(unittest.TestCase):
    def test_same_table_different_trace(self):
        self.assertTrue(identical_iterations(log(ITERS, 0.125), log(ITERS, 0.5)))

    def test_changed_iteration_line(self):
        changed = list(ITERS)
        changed[1] = changed[1].replace("5.25e-04", "5.26e-04")
        self.assertFalse(identical_iterations(log(ITERS, 0.125), log(changed, 0.125)))

    def test_extra_iteration(self):
        more = ITERS + ["   3  1.9887777e+01 4.67e-04 2.41e+03  -1.0 1.58e+02  -0.5 5.33e-02 7.03e-02f  1"]
        self.assertFalse(identical_iterations(log(ITERS, 0.125), log(more, 0.125)))

    def test_restoration_line_with_negative_objective(self):
        # IPOPT prints "%4d%c%14.7e": after "r" a 14-wide negative objective
        # leaves no blank ("2999r-5.0728013e+03 ...").
        line = "2999r-5.0728013e+03 9.49e+03 1.00e+03   1.9 6.96e-06   8.2 1.00e+00 1.00e+00f  1"
        changed = line.replace("6.96e-06", "6.97e-06")
        self.assertEqual(iteration_lines(log([line], 0.1)), [line])
        self.assertFalse(identical_iterations(log([line], 0.1), log([changed], 0.1)))

    def test_lines_extracted(self):
        self.assertEqual(iteration_lines(log(ITERS, 0.125)), ITERS)


if __name__ == "__main__":
    unittest.main()
