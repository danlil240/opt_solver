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

    def test_reanalysis_counts(self):
        # ra=tried/adopted/reverted of the factor's session: adopted > reverted
        # means the factor ran on a re-analysed order.
        text = "\n".join([
            LINE.format(k=1, c=0, d=40, s=0, t=0.5) + " ra=0/0/0",
            LINE.format(k=2, c=0, d=2, s=0, t=0.5) + " ra=1/1/0",
            LINE.format(k=3, c=1, d=4, s=0, t=0.5) + " ra=1/1/0",
            "[smf_ma97] ipm: sessions=1 factorizations=3 accepted=2 aborted=1 reanalyses_tried=1 "
            "reanalyses_adopted=1 reanalyses_reverted=0 reanalyse_s=0.125 flops_avoided=2.500000e+06",
        ])
        r = parse_trace(text)
        self.assertEqual(r["reanalyses_tried"], 1)
        self.assertEqual(r["reanalyses_adopted"], 1)
        self.assertEqual(r["reanalyses_reverted"], 0)
        self.assertAlmostEqual(r["reanalyse_s"], 0.125)
        self.assertEqual(r["factorizations_reanalysed"], 2)
        self.assertEqual(r["delays_reanalysed"], 6)
        self.assertEqual(r["ipm_sessions"], 1)
        self.assertEqual(r["ipm_factorizations"], 3)
        self.assertEqual(r["ipm_accepted"], 2)
        self.assertEqual(r["ipm_aborted"], 1)
        self.assertAlmostEqual(r["flops_avoided"], 2.5e6)

    def test_reverted_probation_factor_counts_as_reanalysed(self):
        # The plugin prints reverted as of the factorization's start: the
        # probation factor that triggers a revert shows 1/1/0, the next 1/1/1.
        text = "\n".join([
            LINE.format(k=1, c=0, d=40, s=0, t=0.5) + " ra=0/0/0",
            LINE.format(k=2, c=0, d=50, s=0, t=0.5) + " ra=1/1/0",
            LINE.format(k=3, c=0, d=40, s=0, t=0.5) + " ra=1/1/1",
        ])
        r = parse_trace(text)
        self.assertEqual(r["factorizations_reanalysed"], 1)
        self.assertEqual(r["delays_reanalysed"], 50)

    def test_no_ipm_summary(self):
        r = parse_trace(LINE.format(k=1, c=0, d=4, s=0, t=0.5))
        self.assertIsNone(r["reanalyses_tried"])
        self.assertIsNone(r["ipm_accepted"])
        self.assertEqual(r["factorizations_reanalysed"], 0)


if __name__ == "__main__":
    unittest.main()
