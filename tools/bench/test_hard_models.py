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


def _ipopt_exe():
    import ipopt_kkt

    return ipopt_kkt.IPOPT_EXE


class TestCops(unittest.TestCase):
    # (builder, size, COPS 3.0 report optimum in the problem's own sense, one_sided).
    # Sizes are the smallest the report tabulates (Dolan, More & Munson,
    # ANL/MCS-TM-273, Tables 1.2-14.2); the report has no nh=100/np=25/nv=25
    # column for chain, rocket, steering, pinene, elec, polygon.
    # one_sided: elec and polygon have many local optima, and the corpus needs
    # local solutions, so assert only that IPOPT is no better than the published
    # optimum in the problem's own sense (min: obj >= ref; max: obj <= ref).
    KNOWN = [
        ("cops_chain", 200, 5.06891, False),          # Table 4.2, min
        ("cops_catmix", 100, -4.80556e-2, False),     # Table 14.2, min
        ("cops_rocket", 400, 1.01283, False),         # Table 10.2, max
        ("cops_steering", 200, 5.54577e-1, False),    # Table 9.2, min
        ("cops_glider", 100, 1.25505e3, False),       # Table 11.2, max
        ("cops_elec", 50, 1.05518e3, True),           # Table 2.2, min
        ("cops_polygon", 50, 7.84015e-1, True),       # Table 1.2, max (best: SNOPT)
        ("cops_pinene", 100, 1.98721e1, False),       # Table 5.2, min
    ]
    # glider needs ~800 IPOPT iterations at nh=100; the others converge well
    # within run_ipopt's default of 200.
    MAX_ITER = {"cops_glider": 3000}

    def test_models_entries(self):
        import hard_models

        names = ["cops_glider_400", "cops_rocket_800", "cops_catmix_800", "cops_chain_800",
                 "cops_pinene_400", "cops_steering_800", "cops_elec_200", "cops_polygon_200"]
        for name in names:
            self.assertIn(name, hard_models.MODELS)
            builder, arg = hard_models.MODELS[name]
            self.assertIs(type(arg), int, name)  # int: --scale applies
            self.assertEqual(builder.__name__, name.rsplit("_", 1)[0])
            self.assertEqual(arg, int(name.rsplit("_", 1)[1]))

    def _check_optimum(self, tmp, name, size, ref, one_sided):
        import pyomo.environ as pe
        import hard_models
        import ipopt_kkt

        model = getattr(hard_models, name)(size)
        sense = next(model.component_data_objects(pe.Objective, active=True)).sense
        nl = pathlib.Path(tmp) / f"{name}_{size}.nl"
        model.write(str(nl), format="nl")
        kw = {"max_iter": self.MAX_ITER[name]} if name in self.MAX_ITER else {}
        res = ipopt_kkt.run_ipopt(nl, "mumps", pathlib.Path(tmp) / name, threads=1, **kw)
        self.assertEqual(res["status"], "Optimal Solution Found.", res)
        # IPOPT's AMPL interface minimizes -f for a maximize objective and
        # prints the internal (minimized) value.
        got = (-1.0 if sense == pe.maximize else 1.0) * res["objective"]
        print(f"  {name}({size}): ref {ref:.6g} got {got:.9g} iters {res['iterations']}", flush=True)
        tol = 1e-4 * abs(ref)
        if not one_sided:
            self.assertLessEqual(abs(got - ref), tol)
        elif sense == pe.maximize:
            self.assertLessEqual(got, ref + tol)      # no larger than the best known maximum
        else:
            self.assertGreaterEqual(got, ref - tol)   # no smaller than the best known minimum

    @unittest.skipUnless(_ipopt_exe().exists(), "IPOPT (smfbench env) not found")
    def test_known_optima(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for name, size, ref, one_sided in self.KNOWN:
                with self.subTest(model=name, size=size):
                    self._check_optimum(tmp, name, size, ref, one_sided)


class TestDefaultSelection(unittest.TestCase):
    ORIGINAL = ["pde_control_120", "pde_control_250", "robot_arm_8k", "obstacle_surface_200",
                "acopf_grid_80", "acopf_grid_160", "portfolio_20k"]

    def test_default_is_original_seven(self):
        import ipopt_kkt

        names = ipopt_kkt.selected_models("")
        self.assertEqual(names, self.ORIGINAL)
        self.assertFalse([n for n in names if n.startswith("pglib_")])

    def test_hard_models_addressable_by_name(self):
        import ipopt_kkt

        self.assertEqual(ipopt_kkt.selected_models("pglib_opf_case118_ieee"), ["pglib_opf_case118_ieee"])
        self.assertIn("pglib_opf_case118_ieee", ipopt_kkt.MODELS)


if __name__ == "__main__":
    unittest.main()
