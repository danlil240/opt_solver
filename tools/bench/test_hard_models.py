"""Tests for the hard-model corpus (tools/bench/hard_models.py).

Run from tools/bench:
  <smfbench>/python.exe -m unittest test_hard_models -v
"""

import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
PGLIB = REPO / "benchmarks" / "fixtures" / "pglib"


class TestPglib(unittest.TestCase):
    @unittest.skipUnless((PGLIB / "pglib_opf_case118_ieee.m").exists(),
                         "run tools/bench/fetch_pglib.py first")
    def test_case118_builds(self):
        import pyomo.environ as pe
        import hard_models

        m = hard_models.acopf_pglib("pglib_opf_case118_ieee")
        n_vars = sum(1 for _ in m.component_data_objects(pe.Var))
        self.assertGreater(n_vars, 500)
        self.assertGreaterEqual(sum(1 for _ in m.component_objects(pe.Constraint)), 1)

    def test_models_shape(self):
        import hard_models

        for name, (builder, arg) in hard_models.MODELS.items():
            self.assertTrue(callable(builder), name)
            self.assertIsInstance(arg, (str, int), name)
        self.assertIn("pglib_opf_case118_ieee", hard_models.MODELS)
        self.assertEqual(len([n for n in hard_models.MODELS if n.startswith("pglib_")]), 8)


if __name__ == "__main__":
    unittest.main()
