"""Tests for the deterministic held-out split in fetch_fronttier.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fetch_fronttier as ff  # noqa: E402


class TestSplit(unittest.TestCase):
    def test_deterministic_and_fraction(self):
        ids = [f"m{i}" for i in range(20)]
        a, b = ff.assign_splits(ids), ff.assign_splits(list(reversed(ids)))
        self.assertEqual(a, b)
        self.assertEqual(sum(v == "heldout" for v in a.values()), 6)

    def test_small_class_keeps_one_heldout(self):
        s = ff.assign_splits(["a", "b", "c"])
        self.assertEqual(sum(v == "heldout" for v in s.values()), 1)

    def test_seed_is_the_task_seed(self):
        self.assertEqual(ff.SEED, 20261027)

    def test_prelabelled_case_keeps_label(self):
        ids = [f"m{i}" for i in range(10)]
        fixed = {"m0": "heldout", "m1": "heldout", "m2": "tuning", "m3": "heldout"}
        s = ff.assign_splits(ids, fixed=fixed)
        for k, v in fixed.items():
            self.assertEqual(s[k], v)
        self.assertEqual(set(s), set(ids))

    def test_every_dhl_name_is_case_or_missing(self):
        import json
        man = json.loads((ff.ROOT / "benchmarks/manifests/fronttier.json").read_text("utf-8"))
        have = {c["matrix_id"] for c in man["cases"]}
        gone = {x["matrix_id"] for x in man["missing"]}
        self.assertEqual(len(ff.DHL_HARD), 24)
        for _, name, _ in ff.DHL_HARD:
            self.assertTrue(name in have or name in gone, name)
        self.assertNotIn("TSOPF_FS_*", gone)
        for c in man["cases"]:
            if c["matrix_id"] in ff.REPORT_ONLY:
                self.assertTrue(c.get("report_only"))


if __name__ == "__main__":
    unittest.main()
