import unittest

from ipm_gates import (gate_abort, gate_outcome, identical_iterations, iteration_lines, merge_repeats,
                       objective_differs, trace_entries)

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



def trace(k, corr, neg, abort=0, skipped=0.0, t=0.01, two=3):
    return (f"[smf_ma97] factor {k}: corr={corr} delays=3 static=0 pos={10 - neg} neg={neg} zero=0 "
            f"L=1000 t={t} scaling=0 scale=in u=1.0e-08 ordering=2(req 5) nemin=8 two={two} "
            f"abort={abort} flops=2.000000e+06 skipped={skipped:.6e}")


# Factor 2 has the wrong inertia (neg 5 > 4) and IPOPT refactors (factor 3,
# corr=1). With the abort, factor 2 stops at a partial count of 5 of its 6.
BASE_TRACE = [trace(1, 0, 4), trace(2, 0, 6), trace(3, 1, 4), trace(4, 0, 4)]
FEAT_TRACE = [trace(1, 0, 4, t=0.02), trace(2, 0, 5, abort=1, skipped=1e6, two=0), trace(3, 1, 4),
              trace(4, 0, 4, t=0.5)]


def run_log(iters, traces, status="Optimal Solution Found."):
    return "\n".join(["This is Ipopt version 3.14.20", HEADER] + iters + [f"EXIT: {status}"] + traces)


def row(model, solver, fact_s, corrections=1):
    return {"model": model, "solver": solver, "status": "Optimal Solution Found.", "iterations": 2,
            "linear_system_factorization_s": fact_s, "inertia_corrections": corrections}


class TestAbortGate(unittest.TestCase):
    def setUp(self):
        self.base = [row("a", "smf", 1.0), row("b", "smf", 0.5, corrections=0)]
        self.feat = [row("a", "smf_abort", 0.9), row("b", "smf_abort", 0.5, corrections=0)]
        self.logs_base = {"a": run_log(ITERS, BASE_TRACE), "b": run_log(ITERS, BASE_TRACE[:1])}
        self.logs_feat = {"a": run_log(ITERS, FEAT_TRACE), "b": run_log(ITERS, BASE_TRACE[:1])}

    def gate(self, **kw):
        return gate_abort(self.base, self.feat, self.logs_base, self.logs_feat, **kw)

    def test_identical_logs_and_ten_percent_saving_pass(self):
        res = self.gate()
        self.assertTrue(res["pass"])
        a = res["models"]["a"]
        self.assertTrue(a["identical"])
        self.assertAlmostEqual(a["fact_saving"], 0.1)
        self.assertEqual(a["aborts"], 1)
        self.assertEqual(a["flops_avoided"], 1e6)
        self.assertAlmostEqual(a["flops_avoided_frac"], 1e6 / 8e6)
        self.assertTrue(res["models"]["b"]["identical"])
        self.assertEqual(res["models"]["b"]["fact_saving"], 0.0)

    def test_one_differing_log_fails(self):
        changed = list(ITERS)
        changed[1] = changed[1].replace("5.25e-04", "5.26e-04")
        self.logs_feat["b"] = run_log(changed, BASE_TRACE[:1])
        res = self.gate()
        self.assertFalse(res["pass"])
        self.assertFalse(res["models"]["b"]["identical"])
        self.assertTrue(res["models"]["a"]["identical"])

    def test_different_exit_status_fails(self):
        self.feat[1]["status"] = "Maximum Number of Iterations Exceeded."
        self.assertFalse(self.gate()["models"]["b"]["identical"])

    def test_small_saving_fails(self):
        self.feat[0]["linear_system_factorization_s"] = 0.97
        res = self.gate()
        self.assertTrue(res["all_identical"])
        self.assertFalse(res["pass"])

    def test_saving_must_be_on_a_tuning_model(self):
        self.assertFalse(self.gate(tuning={"b"})["pass"])
        self.assertTrue(self.gate(tuning={"a"})["pass"])

    def test_saving_of_a_diverged_run_does_not_count(self):
        changed = list(ITERS)
        changed[1] = changed[1].replace("5.25e-04", "5.26e-04")
        self.logs_feat["a"] = run_log(changed, FEAT_TRACE)
        res = self.gate()
        self.assertFalse(res["models"]["a"]["identical"])
        self.assertFalse(res["saving_met"])

    def test_saving_needs_corrections(self):
        self.base[0]["inertia_corrections"] = 0
        self.assertFalse(self.gate()["pass"])

    def test_accepted_factorization_must_match(self):
        feat = list(FEAT_TRACE)
        feat[3] = feat[3].replace("delays=3", "delays=4")
        self.logs_feat["a"] = run_log(ITERS, feat)
        res = self.gate()
        self.assertFalse(res["models"]["a"]["identical"])
        self.assertFalse(res["models"]["a"]["trace_match"])
        self.assertTrue(res["models"]["a"]["iterations_identical"])

    def test_abort_of_an_accepted_factorization_fails(self):
        feat = list(FEAT_TRACE)
        feat[3] = trace(4, 0, 3, abort=1)  # base accepted factor 4 (the last one)
        self.logs_feat["a"] = run_log(ITERS, feat)
        self.assertFalse(self.gate()["models"]["a"]["identical"])

    def test_completed_abort_fails(self):
        self.logs_feat["a"] = run_log(ITERS, FEAT_TRACE + [
            "[smf_ma97] completed aborted factor 2 before a solve: neg=6"])
        self.assertFalse(self.gate()["models"]["a"]["identical"])

    def test_trace_entries_ignore_time(self):
        e = trace_entries("\n".join([trace(1, 0, 4, t=0.1), "noise", trace(2, 1, 4, t=0.2)]))
        self.assertEqual([x["k"] for x in e], [1, 2])
        self.assertEqual(e[0]["key"].replace("factor 1", ""), e[1]["key"].replace("corr=1", "corr=0"))
        self.assertEqual(e[0]["fields"]["abort"], "0")
        self.assertEqual(e[1]["fields"]["neg"], "4")


class TestAbortSolverId(unittest.TestCase):
    def test_smf_abort_is_smf_with_the_abort_on(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_abort"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_abort"), {"SMF_PLUGIN_STATS": "2", "SMF_IPM_ABORT": "1"})
        self.assertNotIn("SMF_IPM_ABORT", ipopt_kkt.solver_env("smf"))


def orow(model, solver, iters, wall, status="Optimal Solution Found."):
    return {"model": model, "solver": solver, "status": status, "iterations": iters, "ipopt_wall_s": wall,
            "linear_system_factorization_s": wall / 2}


class TestOutcomeGate(unittest.TestCase):
    """Feature 2/3 gate: iterations within +-5% (geomean), no new failure,
    wall time not worse (geomean over models with base wall >= 1 s)."""

    def setUp(self):
        self.base = [orow("a", "smf", 100, 10.0), orow("b", "smf", 50, 4.0), orow("c", "smf", 20, 0.1)]

    def feat(self, iters=(100, 50, 20), walls=(9.0, 3.6, 0.09), statuses=None):
        statuses = statuses or ["Optimal Solution Found."] * 3
        return [orow(m, "smf_reanalyse", i, w, st) for m, i, w, st in zip("abc", iters, walls, statuses)]

    def test_iteration_ratio_1_06_fails(self):
        res = gate_outcome(self.base, self.feat(iters=(106, 53, 21.2)), None)
        self.assertAlmostEqual(res["models"]["a"]["iter_ratio"], 1.06)
        self.assertAlmostEqual(res["iter_geomean"], 1.06)
        self.assertFalse(res["pass"])

    def test_new_failure_fails(self):
        res = gate_outcome(self.base, self.feat(statuses=["Optimal Solution Found.",
                                                          "Maximum Number of Iterations Exceeded.",
                                                          "Optimal Solution Found."]), None)
        self.assertTrue(res["models"]["b"]["failed_new"])
        self.assertFalse(res["models"]["a"]["failed_new"])
        self.assertFalse(res["pass"])

    def test_same_iterations_faster_passes(self):
        res = gate_outcome(self.base, self.feat(), None)
        self.assertAlmostEqual(res["models"]["a"]["iter_ratio"], 1.0)
        self.assertAlmostEqual(res["models"]["a"]["wall_ratio"], 0.9)
        self.assertAlmostEqual(res["wall_geomean"], 0.9)
        self.assertTrue(res["pass"])

    def test_wall_mean_skips_models_under_one_second(self):
        res = gate_outcome(self.base, self.feat(walls=(9.0, 3.6, 0.5)), None)
        self.assertAlmostEqual(res["models"]["c"]["wall_ratio"], 5.0)
        self.assertFalse(res["models"]["c"]["in_wall_mean"])
        self.assertAlmostEqual(res["wall_geomean"], 0.9)
        self.assertTrue(res["pass"])

    def test_slower_fails(self):
        res = gate_outcome(self.base, self.feat(walls=(11.0, 4.4, 0.1)), None)
        self.assertAlmostEqual(res["wall_geomean"], 1.1)
        self.assertFalse(res["pass"])

    def test_old_failure_is_not_new(self):
        self.base[0]["status"] = "Maximum Number of Iterations Exceeded."
        res = gate_outcome(self.base, self.feat(statuses=["Maximum Number of Iterations Exceeded."] +
                                                ["Optimal Solution Found."] * 2), None)
        self.assertFalse(res["models"]["a"]["failed_new"])
        self.assertTrue(res["pass"])

    def test_models_selects_and_missing_feature_run_fails(self):
        res = gate_outcome(self.base, self.feat()[:2], ["a", "b"])
        self.assertEqual(sorted(res["models"]), ["a", "b"])
        self.assertTrue(res["pass"])
        res = gate_outcome(self.base, self.feat()[:2], ["a", "c"])
        self.assertTrue(res["models"]["c"]["failed_new"])
        self.assertFalse(res["pass"])


class TestReanalyseSolverId(unittest.TestCase):
    def test_smf_reanalyse_is_smf_with_reanalysis_on(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_reanalyse"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_reanalyse"),
                         {"SMF_PLUGIN_STATS": "2", "SMF_IPM_REANALYSE": "1"})
        self.assertNotIn("SMF_IPM_REANALYSE", ipopt_kkt.solver_env("smf"))

    def test_smf_fallback_is_smf_with_fallback_static_pivoting(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_fallback"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_fallback"),
                         {"SMF_PLUGIN_STATS": "2", "SMF_IPM_STATIC": "fallback"})
        self.assertNotIn("SMF_IPM_STATIC", ipopt_kkt.solver_env("smf"))
        self.assertNotIn("SMF_IPM_STATIC_ZERO", ipopt_kkt.solver_env("smf_fallback"))

    def test_smf_fallback_zero_adds_zero_accounting(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_fallback_zero"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_fallback_zero"),
                         {"SMF_PLUGIN_STATS": "2", "SMF_IPM_STATIC": "fallback", "SMF_IPM_STATIC_ZERO": "1"})

    def test_smf_reg_is_smf_in_regularized_mode(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_reg"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_reg"),
                         {"SMF_PLUGIN_STATS": "2", "SMF_IPM_STATIC": "regularized"})

    def test_smf_reg_zero_adds_zero_accounting(self):
        import ipopt_kkt
        self.assertEqual(ipopt_kkt.solver_options("smf_reg_zero"), ipopt_kkt.solver_options("smf"))
        self.assertEqual(ipopt_kkt.solver_env("smf_reg_zero"),
                         {"SMF_PLUGIN_STATS": "2", "SMF_IPM_STATIC": "regularized", "SMF_IPM_STATIC_ZERO": "1"})


if __name__ == "__main__":
    unittest.main()


class TestMergeRepeats(unittest.TestCase):
    @staticmethod
    def row(model, solver, wall, fact, iters=10, obj=1.0, status="Optimal Solution Found."):
        return {"model": model, "solver": solver, "ipopt_wall_s": wall, "linear_system_factorization_s": fact,
                "linear_system_backsolve_s": fact / 10, "iterations": iters, "objective": obj, "status": status}

    def test_median_of_three_and_runs_kept(self):
        runs = [[self.row("a", "smf", 3.0, 1.0)], [self.row("a", "smf", 1.0, 3.0)], [self.row("a", "smf", 2.0, 2.0)]]
        (m,) = merge_repeats(runs)
        self.assertEqual(m["ipopt_wall_s"], 2.0)
        self.assertEqual(m["linear_system_factorization_s"], 2.0)
        self.assertEqual(m["ipopt_wall_s_runs"], [3.0, 1.0, 2.0])
        self.assertEqual(m["repeats"], 3)
        self.assertTrue(m["repeats_consistent"])

    def test_single_run_rows_pass_through(self):
        (m,) = merge_repeats([[self.row("a", "ma27", 1.5, 0.5)]])
        self.assertEqual(m["ipopt_wall_s"], 1.5)
        self.assertEqual(m["repeats"], 1)

    def test_differing_iterations_are_inconsistent(self):
        runs = [[self.row("a", "smf", 1.0, 1.0, iters=10)], [self.row("a", "smf", 1.0, 1.0, iters=11)]]
        (m,) = merge_repeats(runs)
        self.assertFalse(m["repeats_consistent"])

    def test_rows_grouped_by_model_and_solver_in_first_seen_order(self):
        runs = [[self.row("a", "smf", 1.0, 1.0), self.row("a", "smf_reg", 4.0, 1.0)],
                [self.row("a", "smf", 3.0, 1.0), self.row("a", "smf_reg", 2.0, 1.0)]]
        merged = merge_repeats(runs)
        self.assertEqual([(r["model"], r["solver"]) for r in merged], [("a", "smf"), ("a", "smf_reg")])
        self.assertEqual(merged[0]["ipopt_wall_s"], 2.0)  # median of two = mean
        self.assertEqual(merged[1]["ipopt_wall_s"], 3.0)


class TestObjectiveDiffers(unittest.TestCase):
    def test_relative_tolerance(self):
        self.assertFalse(objective_differs(18438.9617, 18438.9617 * (1 + 5e-7)))
        self.assertTrue(objective_differs(18438.9617, 18439.0937))
        self.assertTrue(objective_differs(-5072.80, -17.85))

    def test_missing_objective_is_not_flagged(self):
        self.assertFalse(objective_differs(None, 1.0))
        self.assertFalse(objective_differs(1.0, None))

    def test_zero_objective_uses_absolute_floor(self):
        self.assertFalse(objective_differs(0.0, 1e-9))
        self.assertTrue(objective_differs(0.0, 1e-3))
