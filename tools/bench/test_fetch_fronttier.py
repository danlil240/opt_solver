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


if __name__ == "__main__":
    unittest.main()
