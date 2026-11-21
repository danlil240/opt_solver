"""Stored solutions, looked up by task features."""

from __future__ import annotations

import pathlib

import numpy as np


class SolutionLibrary:
    """Solutions of past tasks: features, primal x, duals lam_x and lam_g,
    objective. Nearest-neighbour lookup in feature space (Euclidean; the spec
    scales its features)."""

    def __init__(self) -> None:
        self.F: list[np.ndarray] = []
        self.x: list[np.ndarray] = []
        self.lam_x: list[np.ndarray] = []
        self.lam_g: list[np.ndarray] = []
        self.obj: list[float] = []
        self._Fm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.F)

    def add(self, feat, x, lam_x, lam_g, obj: float, min_dist: float = 0.0) -> bool:
        """Store a solution. With min_dist > 0, a solution whose task lies
        within min_dist of a stored one replaces it only when its objective is
        lower; returns whether anything changed."""
        feat = np.asarray(feat, float).ravel()
        if min_dist > 0 and len(self):
            d, j = self.nearest(feat, 1)
            if d[0] < min_dist:
                if obj >= self.obj[j[0]]:
                    return False
                self.F[j[0]], self.x[j[0]], self.lam_x[j[0]], self.lam_g[j[0]], self.obj[j[0]] = (
                    feat, np.asarray(x, float), np.asarray(lam_x, float), np.asarray(lam_g, float), float(obj))
                self._Fm = None
                return True
        self.F.append(feat)
        self.x.append(np.asarray(x, float))
        self.lam_x.append(np.asarray(lam_x, float))
        self.lam_g.append(np.asarray(lam_g, float))
        self.obj.append(float(obj))
        self._Fm = None
        return True

    def nearest(self, feat, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Distances and indices of the k nearest stored tasks, closest first."""
        if not len(self):
            return np.zeros(0), np.zeros(0, int)
        if self._Fm is None:
            self._Fm = np.array(self.F)
        d = np.sqrt(np.sum((self._Fm - np.asarray(feat, float).ravel()) ** 2, axis=1))
        idx = np.argsort(d, kind="stable")[:k]
        return d[idx], idx

    def entry(self, j: int) -> dict:
        return {"x": self.x[j], "lam_x": self.lam_x[j], "lam_g": self.lam_g[j], "obj": self.obj[j]}

    def save(self, path) -> None:
        np.savez_compressed(pathlib.Path(path), F=np.array(self.F), x=np.array(self.x),
                            lam_x=np.array(self.lam_x), lam_g=np.array(self.lam_g), obj=np.array(self.obj))

    @classmethod
    def load(cls, path) -> "SolutionLibrary":
        z = np.load(pathlib.Path(path))
        lib = cls()
        for i in range(len(z["obj"])):
            lib.add(z["F"][i], z["x"][i], z["lam_x"][i], z["lam_g"][i], float(z["obj"][i]))
        return lib
