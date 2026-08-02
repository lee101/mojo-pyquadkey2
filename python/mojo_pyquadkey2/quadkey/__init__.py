"""Drop-in-compatible QuadKey objects and helpers."""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from itertools import product
import re
from typing import Dict, Generator, Iterable, List, Tuple

from . import tilesystem

LAT_STR = "lat"
LON_STR = "lon"
LATITUDE_RANGE = (-85.05112878, 85.05112878)
LONGITUDE_RANGE = (-180.0, 180.0)
LEVEL_RANGE = (1, 29)
KEY_PATTERN = re.compile("^[0-3]+$")


class TileAnchor(IntEnum):
    ANCHOR_NW = 0
    ANCHOR_NE = 1
    ANCHOR_SW = 2
    ANCHOR_SE = 3
    ANCHOR_CENTER = 4


def valid_level(level: int) -> bool:
    try:
        QuadKey.validate_level(level)
        return True
    except ValueError:
        return False


def valid_geo(lat: float, lon: float) -> bool:
    try:
        QuadKey.validate_geo(lat, lon)
        return True
    except ValueError:
        return False


def valid_key(key: str) -> bool:
    try:
        QuadKey.validate_key(key)
        return True
    except ValueError:
        return False


@lru_cache(128)
def minmax_tile(level: int) -> Tuple[int, int]:
    QuadKey.validate_level(level)
    return 0, 2 ** level - 1


class QuadKey:
    def __init__(self, key: str):
        self.validate_key(key)
        self.key = key
        self.tile, self.level = tilesystem.quadkey_to_tile(key)

    def children(self, at_level: int = -1) -> List["QuadKey"]:
        if at_level <= 0:
            at_level = self.level + 1
        if self.level >= LEVEL_RANGE[1] or at_level > LEVEL_RANGE[1] or at_level <= self.level:
            return []
        return [self.__class__(self.key + "".join(part)) for part in product("0123", repeat=at_level - self.level)]

    def parent(self) -> "QuadKey":
        return self.__class__(self.key[:-1])

    def nearby_custom(self, config: Tuple[Iterable[int], Iterable[int]]) -> List[str]:
        maximum = minmax_tile(self.level)[1]
        tiles = {(self.tile[0] + dx, self.tile[1] + dy) for dx, dy in product(config[0], config[1])}
        return sorted(tilesystem.tile_to_quadkey(tile, self.level) for tile in tiles if 0 <= tile[0] <= maximum and 0 <= tile[1] <= maximum)

    def nearby(self, n: int = 1) -> List[str]:
        return self.nearby_custom((range(-n, n + 1), range(-n, n + 1)))

    def is_ancestor(self, node: "QuadKey") -> bool:
        return not (self.level <= node.level or self.key[:len(node.key)] != node.key)

    def is_descendent(self, node: "QuadKey") -> bool:
        return node.is_ancestor(self)

    def side(self) -> float:
        return 256 * tilesystem.ground_resolution(0, self.level)

    def area(self) -> float:
        return self.side() ** 2

    @classmethod
    def xdifference(cls, first: "QuadKey", second: "QuadKey") -> Generator["QuadKey", None, None]:
        assert first.level == second.level
        x, y = 0, 1
        first_tile, second_tile = list(first.tile), list(second.tile)
        se = sw = ne = nw = None
        if first_tile[x] >= second_tile[x] and first_tile[y] <= second_tile[y]:
            ne, sw = first_tile, second_tile
        elif first_tile[x] <= second_tile[x] and first_tile[y] >= second_tile[y]:
            sw, ne = first_tile, second_tile
        elif first_tile[x] <= second_tile[x] and first_tile[y] <= second_tile[y]:
            nw, se = first_tile, second_tile
        else:
            se, nw = first_tile, second_tile
        current = (ne or se)[:]
        while current[x] >= (sw or nw)[x]:
            while (sw is not None and current[y] <= sw[y]) or (nw is not None and current[y] >= nw[y]):
                yield cls.from_tile(tuple(current), first.level)
                current[y] += 1 if sw is not None else -1
            current[x] -= 1
            current[y] = (ne or se)[y]

    def difference(self, to: "QuadKey") -> List["QuadKey"]:
        return list(self.xdifference(self, to))

    @classmethod
    def bbox(cls, quadkeys: List["QuadKey"]) -> List["QuadKey"]:
        assert quadkeys
        level = quadkeys[0].level
        xs, ys = zip(*(item.tile for item in quadkeys))
        return cls.from_tile((max(xs), min(ys)), level).difference(cls.from_tile((min(xs), max(ys)), level))

    def to_tile(self) -> Tuple[Tuple[int, int], int]:
        return self.tile, self.level

    def to_pixel(self, anchor: TileAnchor = TileAnchor.ANCHOR_NW) -> Tuple[int, int]:
        return tilesystem.tile_to_pixel(self.tile, anchor)

    def to_geo(self, anchor: TileAnchor = TileAnchor.ANCHOR_NW) -> Tuple[float, float]:
        return tilesystem.pixel_to_geo(self.to_pixel(anchor), self.level)

    def to_quadint(self) -> int:
        return tilesystem.quadkey_to_quadint(self.key)

    def set_level(self, level: int):
        assert level < self.level
        self.key = self.key[:level]
        self.tile, self.level = tilesystem.quadkey_to_tile(self.key)

    def __eq__(self, other):
        return isinstance(other, QuadKey) and self.key == other.key

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return self.key

    __repr__ = __str__

    def __hash__(self):
        return hash(self.key)

    @classmethod
    def from_tile(cls, tile: Tuple[int, int], level: int) -> "QuadKey":
        return cls(tilesystem.tile_to_quadkey(tile, level))

    @classmethod
    def from_geo(cls, geo: Tuple[float, float], level: int) -> "QuadKey":
        cls.validate_geo(*geo)
        cls.validate_level(level)
        return cls.from_tile(tilesystem.pixel_to_tile(tilesystem.geo_to_pixel(geo, level)), level)

    @staticmethod
    def validate_level(level: int):
        if not LEVEL_RANGE[0] <= level <= LEVEL_RANGE[1]:
            raise ValueError("got invalid zoom level")

    @staticmethod
    def validate_geo(lat: float, lon: float):
        if not LATITUDE_RANGE[0] <= lat <= LATITUDE_RANGE[1] and LONGITUDE_RANGE[0] <= lon <= LONGITUDE_RANGE[1]:
            raise ValueError(f"got invalid lat / lon, bounds are {LATITUDE_RANGE} / {LONGITUDE_RANGE}")

    @staticmethod
    def validate_key(key: str):
        if KEY_PATTERN.match(key) is None:
            raise ValueError("got invalid quadkey")


def from_geo(geo: Tuple[float, float], level: int) -> QuadKey:
    return QuadKey.from_geo(geo, level)


def from_tile(tile: Tuple[int, int], level: int) -> QuadKey:
    return QuadKey.from_tile(tile, level)


def from_str(qk_str: str) -> QuadKey:
    return QuadKey(qk_str)


def from_int(qk_int: int) -> QuadKey:
    return QuadKey(tilesystem.quadint_to_quadkey(qk_int))


def geo_to_dict(geo: Tuple[float, float]) -> Dict[str, float]:
    QuadKey.validate_geo(*geo)
    return {LAT_STR: geo[0], LON_STR: geo[1]}
