# mojo-pyquadkey2

`mojo-pyquadkey2` is a Mojo-backed implementation of batch Web-Mercator and quadint conversions, alongside a compatible Python implementation of pyquadkey2's public `quadkey` API.

## Install and use

This repository uses Pixi to provide Mojo and all Python dependencies.

```bash
pixi install
pixi run build
pixi run test
```

The following command was executed successfully in this checkout:

```bash
pixi run python - <<'PY'
import numpy as np
from mojo_pyquadkey2 import quadkey
from mojo_pyquadkey2.quadkey import tilesystem

tile = quadkey.from_geo((49.014205, 8.420025), 16)
print(tile.key)
print(tile.to_tile())

points = np.array([[49.014205, 8.420025], [-27.052395, 152.97702]])
print(tilesystem.geo_to_pixel_many(points, 16))
PY
```

It prints:

```text
1202032333311320
((34300, 22502), 16)
[[ 8781010  5760658]
 [15517854  9698976]]
```

## Scope and compatibility

The Python `quadkey` module implements the upstream public `QuadKey` and `TileAnchor` API: construction from strings, tiles, geographic coordinates, and quadints; validation; tile, pixel, and geographic conversion; hierarchy and neighbourhood operations; bounding boxes; area and side helpers; and quadint round trips. The parity suite exercises each of these feature groups against pyquadkey2.

The Mojo-specific additions are contiguous NumPy batch helpers in `mojo_pyquadkey2.quadkey.tilesystem`:

- `geo_to_pixel_many(geo, level)`
- `pixel_to_geo_many(pixel, level)`
- `ground_resolution_many(latitude, level)`
- `tile_to_quadint_many(tile, level)`

These helpers require the documented shapes, normalize accepted inputs to native `float64` or `int64` buffers, reject unsafe narrowing and non-finite projection coordinates, and retain every NumPy buffer for the entire native call. They are not part of upstream pyquadkey2. This project does not reproduce upstream's packaging, Cython build, or multiprocessing benchmark harness.

## Benchmark

Fresh output from `pixi run bench` on this machine, using 500,000 items at level 18; values are the best of three runs and include Python and ctypes dispatch.

Machine: Linux-6.8.0-136-generic-x86_64-with-glibc2.39; Python 3.13.14.

| case | mojo-pyquadkey2 | pyquadkey2 | ratio |
|---|---:|---:|---:|
| geo_to_pixel_many | 30.1 ms | 640.3 ms | 21.28x faster |
| pixel_to_geo_many | 31.2 ms | 720.8 ms | 23.09x faster |
| ground_resolution_many | 10.7 ms | 76.3 ms | 7.16x faster |
| tile_to_quadint_many | 12.3 ms | 2315.6 ms | 188.16x faster |

No GPU path is provided. These kernels perform too little work per transferred byte
to amortize device copies and launch overhead at the benchmark size; the CPU is the
default and fastest practical execution path. The profiling baseline also found no
kernel at parity with or slower than upstream, so kernels already more than 5x ahead
were deliberately left unchanged.

## How it works

`src/capi.mojo` compiles into a shared library with a small C ABI. Python passes addresses of validated, contiguous NumPy buffers through `ctypes`; Mojo reconstructs typed pointers and writes only into caller-owned result buffers. Coordinate columns are copied into separate contiguous arrays before the call, so native loops never read a strided column. The object-oriented API remains Python because its per-object work is too small to justify a native boundary crossing.
