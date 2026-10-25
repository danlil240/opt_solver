import json
import pathlib
import tempfile
import unittest
from report import evaluate_report, main as report_main
from run import main as run_main


def make_record(
    matrix_id: str,
    solver_id: str,
    outcome: str = "Success",
    wall_ms: float = 10.0,
    eta_inf: float = 1e-14,
    omega: float = 1e-14,
    finite: bool = True,
    family_id: str = "fam_a",
    sequence_id: str = "",
    trial_id: int = 0,
    warmup: bool = False,
    peak_rss_bytes: int = 1024 * 1024,
):
    return {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "sequence_id": sequence_id or matrix_id,
        "family_id": family_id,
        "solver_id": solver_id,
        "outcome": outcome,
        "wall_ms": wall_ms,
        "peak_rss_bytes": peak_rss_bytes,
        "trial_id": trial_id,
        "warmup": warmup,
        "validation": {
            "finite": finite,
            "eta_inf": eta_inf,
            "omega": omega,
            "relative_residual": eta_inf,
        },
    }


class TestBenchmarkReport(unittest.TestCase):
    def test_unavailable_cannot_win_and_blocks_required_baseline(self):
        records = [
            make_record("m1", "smf", outcome="Success", wall_ms=5.0),
            make_record("m1", "ma97", outcome="Unavailable", wall_ms=0.001, finite=False),
        ]
        summary = evaluate_report(records, gate="G2", required_baselines=["ma97"])
        self.assertFalse(summary["qualified"])
        self.assertIn("missing_required_baseline:ma97", summary["reasons"])
        inst = summary["instances"]["m1"]
        self.assertNotEqual(inst.get("winner"), "ma97")
        self.assertIsNone(inst.get("fastest_baseline_ms"))

    def test_timeout_and_oom_remain_in_denominators(self):
        records = [
            make_record("m1", "smf", outcome="Success", wall_ms=5.0),
            make_record("m2", "smf", outcome="Success", wall_ms=6.0),
            make_record("m3", "smf", outcome="Success", wall_ms=7.0),
            make_record("m1", "ma57", outcome="Success", wall_ms=10.0),
            make_record("m2", "ma57", outcome="Timeout", wall_ms=1000.0, finite=False),
            make_record("m3", "ma57", outcome="OutOfMemory", wall_ms=500.0, finite=False),
        ]
        summary = evaluate_report(records, gate="G0", required_baselines=["ma57"])
        ma57_rel = summary["reliability"]["ma57"]
        self.assertEqual(ma57_rel["total_cases"], 3)
        self.assertEqual(ma57_rel["solved_cases"], 1)
        self.assertAlmostEqual(ma57_rel["solve_rate"], 1.0 / 3.0)
        # Jointly solved is only 1 out of 3 cases -> coverage = 1/3
        self.assertAlmostEqual(summary["coverage"], 1.0 / 3.0)

    def test_no_jointly_solved_cases_produces_insufficient_coverage(self):
        records = [
            make_record("m1", "smf", outcome="Success", wall_ms=5.0),
            make_record("m1", "ma57", outcome="Timeout", wall_ms=100.0, finite=False),
            make_record("m2", "smf", outcome="Failed", wall_ms=5.0, finite=False),
            make_record("m2", "ma57", outcome="Success", wall_ms=8.0),
        ]
        summary = evaluate_report(records, gate="G2", required_baselines=["ma57"])
        self.assertFalse(summary["qualified"])
        self.assertEqual(summary["status"], "insufficient coverage")
        self.assertIsNone(summary["geomean_speedup"])

    def test_smf_failure_or_inaccurate_blocks_qualification(self):
        records = [
            make_record("m1", "smf", outcome="Success", wall_ms=4.0),
            make_record("m1", "ma57", outcome="Success", wall_ms=10.0),
            # Second case returns Success from solver but has excessive backward error:
            make_record(
                "m2",
                "smf",
                outcome="Success",
                wall_ms=3.0,
                eta_inf=1e-4,
                omega=1e-3,
            ),
            make_record("m2", "ma57", outcome="Success", wall_ms=10.0),
        ]
        summary = evaluate_report(records, gate="G1", required_baselines=["ma57"])
        self.assertFalse(summary["qualified"])
        self.assertTrue(any("smf_failure" in r for r in summary["reasons"]))

    def test_family_weights_and_bootstrap_preserve_sequence_grouping(self):
        records = []
        # Family A has 10 correlated iterations in sequence seqA_1, all speedup = 4.0x
        for step in range(10):
            mid = f"seqA_1_step{step}"
            records.append(
                make_record(
                    mid,
                    "smf",
                    outcome="Success",
                    wall_ms=2.5,
                    family_id="fam_A",
                    sequence_id="seqA_1",
                )
            )
            records.append(
                make_record(
                    mid,
                    "ma57",
                    outcome="Success",
                    wall_ms=10.0,
                    family_id="fam_A",
                    sequence_id="seqA_1",
                )
            )
        # Family B has 1 sequence seqB_1 with 1 step, speedup = 1.0x
        records.append(
            make_record(
                "seqB_1_step0",
                "smf",
                outcome="Success",
                wall_ms=10.0,
                family_id="fam_B",
                sequence_id="seqB_1",
            )
        )
        records.append(
            make_record(
                "seqB_1_step0",
                "ma57",
                outcome="Success",
                wall_ms=10.0,
                family_id="fam_B",
                sequence_id="seqB_1",
            )
        )

        summary = evaluate_report(
            records,
            gate="G2",
            required_baselines=["ma57"],
            bootstrap_samples=400,
            seed=123,
        )
        # Equal family weighting: geometric mean of family A (4.0) and family B (1.0) is sqrt(4*1) = 2.0,
        # NOT 4.0^(10/11) * 1.0^(1/11) = 3.52!
        self.assertAlmostEqual(summary["geomean_speedup"], 2.0, places=5)
        # Resampling whole families/sequences with only 2 families (4.0 and 1.0) must include 1.0 at the lower bound:
        self.assertLessEqual(summary["ci95_low"], 1.000001)
        self.assertGreaterEqual(summary["ci95_high"], 3.999999)
        self.assertFalse(summary["qualified"])

    def test_cli_returns_nonzero_on_injected_failure_and_missing_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            jsonl_path = tmp_dir / "records.jsonl"
            records = [
                make_record("m1", "smf", outcome="Success", wall_ms=5.0, eta_inf=1e-3, omega=1e-2),
                make_record("m1", "ma97", outcome="Unavailable", wall_ms=0.0, finite=False),
            ]
            with jsonl_path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            rc_g1 = report_main(["--input", str(jsonl_path), "--gate", "G1"])
            self.assertNotEqual(rc_g1, 0)

            rc_base = report_main(
                ["--input", str(jsonl_path), "--gate", "G0", "--require-baselines", "ma97"]
            )
            self.assertNotEqual(rc_base, 0)

            manifest_path = tmp_dir / "smoke.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [{"matrix_id": "m1", "family_id": "f1", "sequence_id": "s1"}],
                        "solvers": ["smf", "ma97"],
                        "required_baselines": ["ma97"],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = tmp_dir / "out"
            rc_run = run_main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(out_dir),
                    "--require-baselines",
                    "--trials",
                    "10",
                    "--warmups",
                    "1",
                    "--binary",
                    str(tmp_dir / "nonexistent_binary"),
                ]
            )
            self.assertNotEqual(rc_run, 0)
            self.assertTrue((out_dir / "records.jsonl").exists())
            self.assertTrue((out_dir / "environment.json").exists())


if __name__ == "__main__":
    unittest.main()
