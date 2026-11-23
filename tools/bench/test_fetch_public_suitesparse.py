"""Unit tests for tools/bench/fetch_public_suitesparse.py."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fetch_public_suitesparse as fps  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestSeriesSiblingExclusion(unittest.TestCase):
    def test_series_sibling_exclusion(self):
        manifest = json.loads(
            (ROOT / "benchmarks" / "manifests" / "fronttier.json").read_text(encoding="utf-8")
        )
        excluded_names, excluded_stems = fps.fronttier_exclusion_stems(manifest)

        # Exact names from fronttier.json must be excluded.
        for name in ("af_shell7", "c-big", "cont-201", "TSOPF_FS_b300", "ncvxqp9"):
            self.assertIn(name, excluded_names, name)
            self.assertTrue(fps.is_excluded(name, excluded_names, excluded_stems), name)

        # Same-generator series siblings must be excluded.
        for sibling in ("af_shell8", "cont-200", "TSOPF_FS_b162_c2", "nd12k", "thermal3"):
            self.assertTrue(
                fps.is_excluded(sibling, excluded_names, excluded_stems),
                f"expected sibling {sibling} to be excluded",
            )

        # Distinct Schenk_IBMNA problems must be kept (only c-big is excluded).
        for keep in ("c-55", "c-73"):
            self.assertFalse(
                fps.is_excluded(keep, excluded_names, excluded_stems),
                f"expected {keep} to remain eligible",
            )


class TestParseAndSelect(unittest.TestCase):
    def _build_synthetic_ssstats(self) -> str:
        lines = [
            "48",
            "20-Nov-2026 12:00:00",
        ]
        # Excluded entries: graph, combinatorial, binary, non-square, unsymmetric, out-of-range.
        lines.extend([
            "SNAP,roadNet-CA,100000,100000,500000,1,0,0,0,1.0,1.0,undirected graph",
            "DIMACS10,delaunay_n16,65536,65536,393150,1,1,0,0,1.0,1.0,combinatorial problem",
            "Gset,G1,30000,30000,200000,1,1,0,0,1.0,1.0,optimization problem",
            "Suite,rect1,50000,40000,200000,1,0,0,0,1.0,1.0,optimization problem",
            "Suite,unsym1,50000,50000,200000,1,0,0,0,1.0,0.5,optimization problem",
            "Suite,cplx1,50000,50000,200000,0,0,0,0,1.0,1.0,circuit simulation problem",
            "Suite,too_small,19999,19999,100000,1,0,0,0,1.0,1.0,optimization problem",
            "Suite,too_large_n,600001,600001,2000000,1,0,0,1,1.0,1.0,structural problem",
            "Suite,too_large_nnz,100000,100000,30000001,1,0,0,1,1.0,1.0,structural problem",
            "Suite,bipart1,50000,50000,200000,1,0,0,0,1.0,1.0,bipartite graph",
            "Suite,dup1,50000,50000,200000,1,0,0,1,1.0,1.0,duplicate structural problem",
        ])

        # public_indef_kkt: 14 in GroupK1 (cap=10), 4 in GroupK2 -> total 14 <= target 20.
        for i in range(14):
            n = 25000 + i * 15000
            lines.append(
                f"GroupK1,kkt_a_{i:02d},{n},{n},{n * 5},1,0,0,0,1.0,1.0,optimization problem"
            )
        for i in range(4):
            n = 30000 + i * 20000
            lines.append(
                f"GroupK2,kkt_b_{i:02d},{n},{n},{n * 5},1,0,0,0,1.0,1.0,power network problem"
            )

        # public_indef_pde: 6 in GroupP1 (cap=3), 5 in GroupP2 (cap=3), 4 in GroupP3 (cap=3).
        for g_idx, (grp, count) in enumerate([("GroupP1", 6), ("GroupP2", 5), ("GroupP3", 4)]):
            for i in range(count):
                n = 22000 + (g_idx * 10 + i) * 12000
                lines.append(
                    f"{grp},pde_{g_idx}_{i},{n},{n},{n * 7},1,0,0,0,1.0,1.0,2D/3D problem"
                )

        # public_spd: 6 groups x 5 matrices each (cap=3 -> 18 <= target 25), plus 2 more groups x 5 -> 24, plus 1 group x 5 -> 27 capped at 25.
        for g_idx in range(9):
            grp = f"GroupS{g_idx}"
            for i in range(5):
                n = 21000 + (g_idx * 5 + i) * 10000
                lines.append(
                    f"{grp},spd_{g_idx}_{i},{n},{n},{n * 8},1,0,0,1,1.0,1.0,structural problem"
                )

        return "\n".join(lines) + "\n"

    def test_parse_and_select_caps_and_order(self):
        csv_text = self._build_synthetic_ssstats()
        entries = fps.parse_ssstats(csv_text)
        self.assertGreater(len(entries), 30)

        selected = fps.select_candidates(entries, set(), set())

        dropped_names = {
            "roadNet-CA", "delaunay_n16", "G1", "rect1", "unsym1",
            "cplx1", "too_small", "too_large_n", "too_large_nnz", "bipart1", "dup1",
        }
        all_picked_names = {
            e["name"] for cls_list in selected.values() for e in cls_list
        }
        self.assertTrue(all_picked_names.isdisjoint(dropped_names))

        # Check per-group caps and target counts.
        kkt = selected["public_indef_kkt"]
        self.assertEqual(len(kkt), 14)  # 10 (GroupK1 capped) + 4 (GroupK2)
        self.assertLessEqual(sum(1 for e in kkt if e["group"] == "GroupK1"), 10)
        self.assertLessEqual(sum(1 for e in kkt if e["group"] == "GroupK2"), 10)

        pde = selected["public_indef_pde"]
        self.assertEqual(len(pde), 9)  # 3 + 3 + 3
        for grp in ("GroupP1", "GroupP2", "GroupP3"):
            self.assertEqual(sum(1 for e in pde if e["group"] == grp), 3)

        spd = selected["public_spd"]
        self.assertEqual(len(spd), 25)  # target count is 25
        for g_idx in range(9):
            self.assertLessEqual(sum(1 for e in spd if e["group"] == f"GroupS{g_idx}"), 3)

        # Determinism: reversed input order produces the exact same selection & order.
        entries_rev = list(reversed(entries))
        selected_rev = fps.select_candidates(entries_rev, set(), set())
        for cls in ("public_indef_kkt", "public_indef_pde", "public_spd"):
            self.assertEqual(
                [e["name"] for e in selected[cls]],
                [e["name"] for e in selected_rev[cls]],
            )


class TestAlgebraicSingularityCheck(unittest.TestCase):
    def _write_mtx(self, directory: pathlib.Path, filename: str, content: str) -> pathlib.Path:
        p = directory / filename
        p.write_text(content, encoding="ascii")
        return p

    def test_algebraic_singularity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = pathlib.Path(tmp)

            # (a) Nonsingular 3x3 saddle-point matrix:
            # [2 0 1; 0 3 0; 1 0 0]
            nonsing = self._write_mtx(
                td,
                "nonsing.mtx",
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "3 3 4\n"
                "1 1 2.0\n"
                "2 2 3.0\n"
                "3 1 1.0\n"
                "3 3 0.0\n",
            )
            is_sing, reason = fps.check_algebraic_singularity(nonsing)
            self.assertFalse(is_sing)
            self.assertEqual(reason, "")

            # (b) Explicit 0.0 entry creating an all-zero row (row 2) after dropping zeros:
            zero_row = self._write_mtx(
                td,
                "zero_row.mtx",
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "3 3 3\n"
                "1 1 2.0\n"
                "2 2 0.0\n"
                "3 3 4.0\n",
            )
            is_sing, reason = fps.check_algebraic_singularity(zero_row)
            self.assertTrue(is_sing)
            self.assertTrue(reason)

            # (c) Unpinned saddle-point matrix with zero-diagonal rows z={3,4}
            # where A[:, z] * 1 == 0 (exact constant-pressure nullspace):
            # B^T = [[1, -1], [-1, 1]] on rows 1..2 vs cols 3..4
            nullspace = self._write_mtx(
                td,
                "nullspace.mtx",
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "4 4 6\n"
                "1 1 2.0\n"
                "2 2 3.0\n"
                "3 1 1.0\n"
                "4 1 -1.0\n"
                "3 2 -1.0\n"
                "4 2 1.0\n",
            )
            is_sing, reason = fps.check_algebraic_singularity(nullspace)
            self.assertTrue(is_sing)
            self.assertTrue(reason)

            # (d) Header comment checks: "nonsingular" must NOT trigger, "singular" MUST trigger.
            nonsing_comment = self._write_mtx(
                td,
                "nonsing_comment.mtx",
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "% Note: this matrix is nonsingular\n"
                "2 2 2\n"
                "1 1 2.0\n"
                "2 2 3.0\n\n\n",
            )
            self.assertEqual(fps.check_algebraic_singularity(nonsing_comment), (False, ""))

            sing_comment = self._write_mtx(
                td,
                "sing_comment.mtx",
                "%%MatrixMarket matrix coordinate real symmetric\n"
                "% Warning: matrix is rank deficient / singular\n"
                "2 2 2\n"
                "1 1 2.0\n"
                "2 2 3.0\n",
            )
            is_sing, reason = fps.check_algebraic_singularity(sing_comment)
            self.assertTrue(is_sing)
            self.assertTrue(reason)


class TestManifestCaseTracksAndReportOnly(unittest.TestCase):
    def test_manifest_case_tracks_and_report_only(self):
        fetched_items = [
            {
                "group": "G1",
                "name": "indef_ok",
                "cls": "public_indef_kkt",
                "matrix_type": "indefinite",
                "mtx_path": "benchmarks/fixtures/public_suitesparse/indef_ok/indef_ok.mtx",
                "matrix_sha256": "a" * 64,
                "n": 25000,
                "nnz_lower": 100000,
                "report_only": False,
            },
            {
                "group": "G2",
                "name": "indef_sing",
                "cls": "public_indef_pde",
                "matrix_type": "indefinite",
                "mtx_path": "benchmarks/fixtures/public_suitesparse/indef_sing/indef_sing.mtx",
                "matrix_sha256": "b" * 64,
                "n": 30000,
                "nnz_lower": 150000,
                "report_only": True,
            },
            {
                "group": "G3",
                "name": "spd_ok",
                "cls": "public_spd",
                "matrix_type": "spd",
                "mtx_path": "benchmarks/fixtures/public_suitesparse/spd_ok/spd_ok.mtx",
                "matrix_sha256": "c" * 64,
                "n": 40000,
                "nnz_lower": 200000,
                "report_only": False,
            },
        ]
        cases = fps.build_manifest_cases(fetched_items)
        self.assertEqual(len(cases), 5)  # 2 + 2 + 1

        by_key = {(c["matrix_id"], c["track"]): c for c in cases}
        self.assertIn(("indef_ok", "ipopt"), by_key)
        self.assertIn(("indef_ok", "recommended"), by_key)
        self.assertIn(("indef_sing", "ipopt"), by_key)
        self.assertIn(("indef_sing", "recommended"), by_key)
        self.assertIn(("spd_ok", "recommended"), by_key)
        self.assertNotIn(("spd_ok", "ipopt"), by_key)

        for key, c in by_key.items():
            self.assertEqual(c["split"], "sealed_public", key)
            self.assertEqual(c["mode"], "cold", key)
            self.assertEqual(c["expected_outcome"], "Success", key)

        self.assertNotIn("report_only", by_key[("indef_ok", "ipopt")])
        self.assertNotIn("report_only", by_key[("indef_ok", "recommended")])
        self.assertTrue(by_key[("indef_sing", "ipopt")]["report_only"])
        self.assertTrue(by_key[("indef_sing", "recommended")]["report_only"])
        self.assertEqual(by_key[("spd_ok", "recommended")]["source"], "SuiteSparse G3/spd_ok")


class TestOfflineFetchAndReplacement(unittest.TestCase):
    def test_fetch_replaces_failed_download_deterministically(self):
        import io
        import tarfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            td = pathlib.Path(tmp)
            ssstats_path = td / "ssstats.csv"
            ssstats_path.write_text(
                "3\n"
                "20-Nov-2026 12:00:00\n"
                "G1,m_fail,20000,20000,60000,1,0,0,1,1.0,1.0,structural problem\n"
                "G2,m_ok,25000,25000,75000,1,0,0,1,1.0,1.0,structural problem\n"
                "G3,m_backup,30000,30000,90000,1,0,0,1,1.0,1.0,structural problem\n",
                encoding="utf-8",
            )
            fronttier_path = td / "fronttier.json"
            fronttier_path.write_text('{"cases": []}', encoding="utf-8")
            out_dir = td / "fixtures"
            manifest_path = td / "public_suitesparse.json"

            mtx_bytes = (
                b"%%MatrixMarket matrix coordinate real symmetric\n"
                b"2 2 2\n1 1 4.0\n2 2 5.0\n"
            )

            def fake_download(group: str, name: str, dest: pathlib.Path) -> str:
                if name == "m_fail":
                    raise RuntimeError("simulated 404")
                with tarfile.open(dest, "w:gz") as tf:
                    info = tarfile.TarInfo(f"{name}/{name}.mtx")
                    info.size = len(mtx_bytes)
                    tf.addfile(info, io.BytesIO(mtx_bytes))
                return f"https://example.invalid/{group}/{name}.tar.gz"

            small_targets = {
                "public_indef_kkt": (0, 10),
                "public_indef_pde": (0, 3),
                "public_spd": (2, 3),
            }
            with (
                mock.patch.object(fps, "TARGETS", small_targets),
                mock.patch.object(fps, "_archive_size", return_value=1024),
                mock.patch.object(fps.fs, "download", side_effect=fake_download),
            ):
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    # Dry-run writes nothing.
                    rc_dry = fps.main([
                        "--ssstats", str(ssstats_path),
                        "--fronttier", str(fronttier_path),
                        "--out", str(out_dir),
                        "--manifest", str(manifest_path),
                    ])
                    self.assertEqual(rc_dry, 0)
                    self.assertFalse(manifest_path.exists())

                    # Fetch skips m_fail and pulls deterministic replacement m_backup.
                    rc_fetch = fps.main([
                        "--fetch",
                        "--ssstats", str(ssstats_path),
                        "--fronttier", str(fronttier_path),
                        "--out", str(out_dir),
                        "--manifest", str(manifest_path),
                        "--budget-mb", "10",
                    ])
                    self.assertEqual(rc_fetch, 0)
                    self.assertTrue(manifest_path.exists())

            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(man["suite"], "public_suitesparse")
            self.assertEqual(man["solvers"], fps.SOLVERS)
            self.assertEqual(man["required_baselines"], [])
            picked = [c["matrix_id"] for c in man["cases"]]
            self.assertEqual(picked, ["m_backup", "m_ok"])
            self.assertEqual(len(man["selection_policy"]["skipped_replacements"]), 1)


if __name__ == "__main__":
    unittest.main()
