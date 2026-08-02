"""ctypes loader for the Mojo projection kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-pyquadkey2.so")
SRC = os.path.join(ROOT, "src", "capi.mojo")
I = ctypes.c_int64
P = ctypes.c_void_p

_SIGNATURES = {
    "mpq_geo_to_pixel": ([P, P, P, P, I, I], None),
    "mpq_pixel_to_geo": ([P, P, P, P, I, I], None),
    "mpq_ground_resolution": ([P, P, I, I], None),
    "mpq_tile_to_quadint": ([P, P, P, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo is not on PATH; run this package through `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    try:
        proc = subprocess.run(
            [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
            capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise BuildError("Mojo build timed out after 1800 seconds") from exc
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def address(values: np.ndarray) -> int:
    return values.ctypes.data
