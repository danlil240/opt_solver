"""Tests for the Phase 0 exit report (tools/bench/phase0_report.py).

Run from tools/bench:
  <smfbench>/python.exe -m unittest test_phase0_report -v
"""

import unittest


def _res(model, fact, corr=0, delays=0, solver="smf", **kw):
    return {"model": model, "solver": solver, "factorizations": fact,
            "inertia_corrections": corr, "delays_total": delays, **kw}


class TestExit(unittest.TestCase):
    def test_three_tuning_models_qualify(self):
        import phase0_report

        results = [
            _res("tuning_a", 10, corr=2),                 # corrections 2/10 = 0.20
            _res("tuning_b", 100, delays=300),            # 3.0 >= 0.01 * 200 = 2.0
            _res("tuning_c", 50, delays=100),             # 2.0 >= 0.01 * 100 = 1.0
            _res("tuning_d", 100, corr=5, delays=10),     # 0.05 < 0.10, 0.1 < 0.01*500
        ]
        n_by_model = {"tuning_a": 1000, "tuning_b": 200, "tuning_c": 100, "tuning_d": 500}
        ok, names = phase0_report.exit_ok(results, n_by_model)
        self.assertTrue(ok)
        self.assertEqual(sorted(names), ["tuning_a", "tuning_b", "tuning_c"])

    def test_heldout_model_is_not_counted(self):
        import phase0_report

        results = [
            _res("tuning_a", 10, corr=2),
            _res("tuning_b", 100, delays=300),
            _res("tuning_c", 50, delays=100, split="heldout"),
        ]
        n_by_model = {"tuning_a": 1000, "tuning_b": 200, "tuning_c": 100}
        ok, names = phase0_report.exit_ok(results, n_by_model)
        self.assertFalse(ok)
        self.assertEqual(sorted(names), ["tuning_a", "tuning_b"])

    def test_heldout_set_membership_and_non_smf_ignored(self):
        import phase0_report

        results = [
            _res("pglib_opf_case300_ieee", 10, corr=5),   # in HELDOUT
            _res("tuning_a", 10, corr=5, solver="ma27", factorizations_none=True),
        ]
        results[1]["factorizations"] = None
        ok, names = phase0_report.exit_ok(results, {"pglib_opf_case300_ieee": 100, "tuning_a": 100})
        self.assertFalse(ok)
        self.assertEqual(names, [])


class TestSuiteOption(unittest.TestCase):
    def test_suite_hard_selects_all_hard_models(self):
        import hard_models
        import ipopt_kkt

        self.assertEqual(ipopt_kkt.selected_models("", "hard"), list(hard_models.MODELS))
        self.assertEqual(ipopt_kkt.selected_models("", ""), list(ipopt_kkt.DEFAULT_MODELS))
        self.assertEqual(ipopt_kkt.selected_models("a,b", ""), ["a", "b"])

    def test_max_iter_and_heldout(self):
        import hard_models
        import ipopt_kkt

        self.assertEqual(ipopt_kkt.max_iter_for("cops_glider_400"), 3000)
        self.assertEqual(ipopt_kkt.max_iter_for("pde_control_120"), 200)
        self.assertEqual(len(hard_models.HELDOUT), 6)
        self.assertTrue(hard_models.HELDOUT <= set(hard_models.MODELS))


if __name__ == "__main__":
    unittest.main()
