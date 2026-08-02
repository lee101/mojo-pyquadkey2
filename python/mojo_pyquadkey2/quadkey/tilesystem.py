"""Tile-system functions plus NumPy batch variants backed by Mojo."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Tuple

import numpy as np

from .._lib import address, f64, lib

EARTH_RADIUS = 6378137
LATITUDE_RANGE = (-85.05112878, 85.05112878)
LONGITUDE_RANGE = (-180.0, 180.0)


def clip(value: float, bounds: Tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def map_size(level: int) -> int:
    return 256 << level


def ground_resolution(lat: float, level: int) -> float:
    return math.cos(clip(lat, LATITUDE_RANGE) * math.pi / 180) * 2 * math.pi * EARTH_RADIUS / map_size(level)


def geo_to_pixel(geo: Tuple[float, float], level: int) -> Tuple[int, int]:
    lat, lon = geo
    lat, lon = clip(lat, LATITUDE_RANGE), clip(lon, LONGITUDE_RANGE)
    x = (lon + 180) / 360
    sin_lat = math.sin(lat * math.pi / 180)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)
    size = map_size(level)
    return int(clip(x * size + 0.5, (0, size - 1))), int(clip(y * size + 0.5, (0, size - 1)))


def pixel_to_geo(pixel: Tuple[float, float], level: int) -> Tuple[float, float]:
    size = map_size(level)
    x = clip(pixel[0], (0, size - 1)) / size - 0.5
    y = 0.5 - clip(pixel[1], (0, size - 1)) / size
    lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    return int(lat * 1e12) / 1e12, int(360 * x * 1e12) / 1e12


def pixel_to_tile(pixel: Tuple[int, int]) -> Tuple[int, int]:
    return pixel[0] // 256, pixel[1] // 256


def tile_to_pixel(tile: Tuple[int, int], anchor=0) -> Tuple[int, int]:
    x, y = tile[0] * 256, tile[1] * 256
    if anchor == 4:
        return x + 128, y + 128
    if anchor == 1:
        return x + 256, y
    if anchor == 2:
        return x, y + 256
    if anchor == 3:
        return x + 256, y + 256
    return x, y


def tile_to_quadkey(tile: Tuple[int, int], level: int) -> str:
    x, y = tile
    chars = []
    for bit in range(level, 0, -1):
        mask = 1 << (bit - 1)
        chars.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(chars)


def quadkey_to_tile(quadkey: str) -> Tuple[Tuple[int, int], int]:
    x = y = 0
    level = len(quadkey)
    for index, char in enumerate(quadkey):
        mask = 1 << (level - index - 1)
        if char in "13":
            x |= mask
        if char in "23":
            y |= mask
    return (x, y), level


def quadint_to_quadkey(quadint: int) -> str:
    level = quadint & 0b11111
    return "".join(str((quadint >> (64 - (index + 1) * 2)) & 3) for index in range(level))


def quadkey_to_quadint(quadkey: str) -> int:
    value = len(quadkey)
    for index, char in enumerate(quadkey):
        value |= int(char) << (64 - (index + 1) * 2)
    return value


def _coordinates(values, name: str, dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if dtype is np.float64:
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numbers")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values")
    elif array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer values")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise OverflowError(f"{name} contains values outside int64 range")
    array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _level(level: int) -> int:
    if not isinstance(level, Integral) or isinstance(level, bool) or not 1 <= level <= 29:
        raise ValueError("level must be an integer from 1 through 29")
    return int(level)


def geo_to_pixel_many(geo, level: int) -> np.ndarray:
    """Project an ``(n, 2)`` latitude/longitude array to int64 pixels in Mojo."""
    level = _level(level)
    source = _coordinates(geo, "geo", np.float64)
    result = np.empty(source.shape, dtype=np.int64)
    # Columns of a C-order (n, 2) matrix are strided; give Mojo contiguous buffers.
    lat, lon = np.ascontiguousarray(source[:, 0]), np.ascontiguousarray(source[:, 1])
    x, y = np.empty(len(source), dtype=np.int64), np.empty(len(source), dtype=np.int64)
    if not len(source):
        return result
    lib().mpq_geo_to_pixel(address(lat), address(lon), address(x), address(y), len(source), level)
    result[:, 0], result[:, 1] = x, y
    return result


def pixel_to_geo_many(pixel, level: int) -> np.ndarray:
    """Inverse-project an ``(n, 2)`` pixel array in Mojo."""
    level = _level(level)
    source = _coordinates(pixel, "pixel", np.int64)
    x, y = np.ascontiguousarray(source[:, 0]), np.ascontiguousarray(source[:, 1])
    lat, lon = np.empty(len(source)), np.empty(len(source))
    if not len(source):
        return np.empty((0, 2), dtype=np.float64)
    lib().mpq_pixel_to_geo(address(x), address(y), address(lat), address(lon), len(source), level)
    return np.column_stack((lat, lon))


def ground_resolution_many(latitude, level: int) -> np.ndarray:
    level = _level(level)
    lat = np.asarray(latitude)
    if lat.ndim != 1:
        raise ValueError("latitude must be one-dimensional")
    if lat.dtype.kind not in "iuf":
        raise TypeError("latitude must contain real numbers")
    lat = f64(lat)
    result = np.empty_like(lat)
    if not len(lat):
        return result
    lib().mpq_ground_resolution(address(lat), address(result), len(lat), level)
    return result


def tile_to_quadint_many(tile, level: int) -> np.ndarray:
    """Encode an ``(n, 2)`` tile array to uint64 binary quadkeys in Mojo."""
    level = _level(level)
    source = _coordinates(tile, "tile", np.int64)
    x, y = np.ascontiguousarray(source[:, 0]), np.ascontiguousarray(source[:, 1])
    result = np.empty(len(source), dtype=np.uint64)
    if not len(source):
        return result
    lib().mpq_tile_to_quadint(address(x), address(y), address(result), len(source), level)
    return result
