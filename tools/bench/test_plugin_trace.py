import unittest

from plugin_trace import parse_trace

LINE = ("[smf_ma97] factor {k}: corr={c} delays={d} static={s} pos=5 neg=3 zero=0 "
        "L=1000 t={t}")


class TestParse(unittest.TestCase):
    def test_counts_corrections(self):
        text = "\n".join([
            "This is Ipopt version 3.14",
            LINE.format(k=1, c=0, d=10, s=0, t=0.5),
            "iter    objective    inf_pr",
            LINE.format(k=2, c=1, d=20, s=1, t=0.25),
            "   1  1.0e+00 1.0e-01",
            LINE.format(k=3, c=0, d=0, s=0, t=0.25) + " scaling=1 ordering=5(req 5)",
            "EXIT: Optimal Solution Found.",
        ])
        r = parse_trace(text)
        self.assertEqual(r["factorizations"], 3)
        self.assertEqual(r["inertia_corrections"], 1)
        self.assertEqual(r["delays_total"], 30)
        self.assertEqual(r["static_total"], 1)
        self.assertAlmostEqual(r["factor_s"], 1.0)

    def test_empty(self):
        r = parse_trace("no trace here")
        self.assertEqual(r["factorizations"], 0)
        self.assertEqual(r["inertia_corrections"], 0)


if __name__ == "__main__":
    unittest.main()
