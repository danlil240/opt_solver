"""Unit tests for the smf_solver Python packaging helper."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import smf_solver


class LoadCLibWindowsDllDirectoryTest(unittest.TestCase):
    def test_bare_dll_name_does_not_call_add_dll_directory_dot(self) -> None:
        added_dirs: list[str] = []

        class _DummyHandle:
            def __enter__(self) -> "_DummyHandle":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def fake_add_dll_directory(path: str) -> _DummyHandle:
            added_dirs.append(path)
            return _DummyHandle()

        fake_cdll = mock.MagicMock()

        with (
            mock.patch.object(smf_solver, "c_library_path", return_value="smf_c.dll"),
            mock.patch.object(smf_solver.sys, "platform", "win32"),
            mock.patch.object(smf_solver.os, "add_dll_directory", side_effect=fake_add_dll_directory, create=True),
            mock.patch.object(smf_solver.ctypes, "CDLL", return_value=fake_cdll) as cdll_mock,
        ):
            smf_solver._load_c_lib()

        self.assertEqual(
            added_dirs,
            [],
            f"bare DLL fallback must not call os.add_dll_directory (got {added_dirs!r})",
        )
        cdll_mock.assert_called_once_with("smf_c.dll")

    def test_explicit_path_adds_resolved_parent_directory(self) -> None:
        added_dirs: list[str] = []
        closed = False

        class _DummyHandle:
            def __enter__(self) -> "_DummyHandle":
                return self

            def __exit__(self, *args: object) -> None:
                nonlocal closed
                closed = True

        def fake_add_dll_directory(path: str) -> _DummyHandle:
            added_dirs.append(path)
            return _DummyHandle()

        fake_cdll = mock.MagicMock()
        this_dir = pathlib.Path(__file__).resolve().parent
        dll_path = str(this_dir / "smf_c.dll")

        with (
            mock.patch.object(smf_solver, "c_library_path", return_value=dll_path),
            mock.patch.object(smf_solver.sys, "platform", "win32"),
            mock.patch.object(smf_solver.os, "add_dll_directory", side_effect=fake_add_dll_directory, create=True),
            mock.patch.object(smf_solver.ctypes, "CDLL", return_value=fake_cdll),
        ):
            smf_solver._load_c_lib()

        self.assertEqual(added_dirs, [str(this_dir)])
        self.assertTrue(closed, "add_dll_directory handle should be closed after loading")


if __name__ == "__main__":
    unittest.main()
