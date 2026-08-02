"""Benchmark the Mojo batch kernels against pyquadkey2's Cython scalar API."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojo_pyquadkey2.quadkey import tilesystem as mojo  # noqa: E402
from pyquadkey2.quadkey import tilesystem as upstream  # noqa: E402


def best_time(fn, repeat: int = 3) -> float:
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        result = min(result, time.perf_counter() - start)
    return result


def measure(name, mojo_fn, upstream_fn):
    mojo_fn()
    ours, theirs = best_time(mojo_fn), best_time(upstream_fn)
    label = "faster" if ours < theirs else "slower"
    print(f"| {name} | {ours * 1e3:.1f} ms | {theirs * 1e3:.1f} ms | {theirs / ours:.2f}x {label} |")


def main():
    rng = np.random.default_rng(42)
    n, level = 500_000, 18
    geo = np.column_stack((rng.uniform(-85, 85, n), rng.uniform(-180, 180, n)))
    pixels = np.array([upstream.geo_to_pixel(tuple(row), level) for row in geo], dtype=np.int64)
    tiles = pixels // 256
    print(f"Machine: {platform.platform()} | Python: {platform.python_version()} | n={n:,}, level={level}")
    print("| case | mojo-pyquadkey2 | pyquadkey2 | ratio |")
    print("|---|---:|---:|---:|")
    measure("geo_to_pixel_many", lambda: mojo.geo_to_pixel_many(geo, level), lambda: [upstream.geo_to_pixel(tuple(row), level) for row in geo])
    measure("pixel_to_geo_many", lambda: mojo.pixel_to_geo_many(pixels, level), lambda: [upstream.pixel_to_geo(tuple(row), level) for row in pixels])
    measure("ground_resolution_many", lambda: mojo.ground_resolution_many(geo[:, 0], level), lambda: [upstream.ground_resolution(lat, level) for lat in geo[:, 0]])
    measure("tile_to_quadint_many", lambda: mojo.tile_to_quadint_many(tiles, level), lambda: [upstream.quadkey_to_quadint(upstream.tile_to_quadkey(tuple(row), level)) for row in tiles])


if __name__ == "__main__":
    main()
