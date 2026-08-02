"""Behavioural parity checks against the published pyquadkey2 package."""

from __future__ import annotations

from operator import attrgetter

import numpy as np
import pytest

from mojo_pyquadkey2 import quadkey as ours
from mojo_pyquadkey2.quadkey import tilesystem as ours_tiles
from pyquadkey2 import quadkey as upstream
from pyquadkey2.quadkey import tilesystem as upstream_tiles


@pytest.mark.parametrize("key", ["0", "0231010", "012301230123012", "3" * 29])
def test_quadkey_tile_and_integer_roundtrips(key):
    left, right = ours.from_str(key), upstream.from_str(key)
    assert left.key == right.key
    assert left.to_tile() == right.to_tile()
    assert left.to_quadint() == right.to_quadint()
    assert ours.from_int(left.to_quadint()).key == key


@pytest.mark.parametrize("geo, level", [((49.014205, 8.420025), 16), ((-27.052395, 152.97702), 6), ((40.0, -105.0), 7), ((85.05112878, 180.0), 29)])
def test_from_geo_and_scalar_tilesystem_parity(geo, level):
    assert ours.from_geo(geo, level).key == upstream.from_geo(geo, level).key
    pixel = ours_tiles.geo_to_pixel(geo, level)
    assert pixel == upstream_tiles.geo_to_pixel(geo, level)
    assert ours_tiles.pixel_to_geo(pixel, level) == upstream_tiles.pixel_to_geo(pixel, level)
    assert ours_tiles.ground_resolution(geo[0], level) == pytest.approx(upstream_tiles.ground_resolution(geo[0], level))


def test_anchors_children_neighbors_and_geometric_helpers():
    left, right = ours.from_str("033"), upstream.from_str("033")
    for anchor in ours.TileAnchor:
        assert left.to_pixel(anchor) == right.to_pixel(upstream.TileAnchor(anchor.value))
        assert left.to_geo(anchor) == right.to_geo(upstream.TileAnchor(anchor.value))
    assert [item.key for item in left.children(at_level=5)] == [item.key for item in right.children(at_level=5)]
    assert left.nearby(2) == right.nearby(2)
    target_left, target_right = ours.from_str("003"), upstream.from_str("003")
    assert {item.key for item in left.difference(target_left)} == {item.key for item in right.difference(target_right)}
    assert {item.key for item in ours.QuadKey.bbox([left, target_left])} == {item.key for item in upstream.QuadKey.bbox([right, target_right])}
    assert left.side() == pytest.approx(right.side())
    assert left.area() == pytest.approx(right.area())


def test_top_level_helpers_and_relationships_parity():
    left, right = ours.from_tile((26, 48), 7), upstream.from_tile((26, 48), 7)
    assert left.key == right.key == "0231010"
    assert left.parent().key == right.parent().key
    assert left.is_ancestor(ours.from_str("023")) == right.is_ancestor(upstream.from_str("023"))
    assert left.is_descendent(ours.from_str("02310101")) == right.is_descendent(upstream.from_str("02310101"))
    assert ours.minmax_tile(7) == upstream.minmax_tile(7)
    assert ours.geo_to_dict((40.0, -105.0)) == upstream.geo_to_dict((40.0, -105.0))
    left.set_level(3)
    right.set_level(3)
    assert left.key == right.key and left.to_tile() == right.to_tile()


@pytest.mark.parametrize("bad", ["", "0156510012", "a"])
def test_validation_and_edge_behavior(bad):
    for module in (ours, upstream):
        with pytest.raises(ValueError):
            module.from_str(bad)
    assert ours.valid_level(29) and not ours.valid_level(30)
    assert ours.valid_geo(0.0, 0.0) and not ours.valid_geo(90.0, 0.0)


def test_mojo_batch_projection_parity():
    rng = np.random.default_rng(7)
    geo = np.column_stack((rng.uniform(-85.05112878, 85.05112878, 20_000), rng.uniform(-180, 180, 20_000)))
    level = 18
    actual_pixels = ours_tiles.geo_to_pixel_many(geo, level)
    expected_pixels = np.array([upstream_tiles.geo_to_pixel(tuple(row), level) for row in geo], dtype=np.int64)
    # Mojo and CPython's Cython build use different libm lowering. At values
    # exactly on a pixel rounding boundary that can move one projected pixel.
    assert np.max(np.abs(actual_pixels - expected_pixels)) <= 1
    actual_geo = ours_tiles.pixel_to_geo_many(actual_pixels, level)
    expected_geo = np.array([upstream_tiles.pixel_to_geo(tuple(row), level) for row in actual_pixels])
    assert np.allclose(actual_geo, expected_geo, atol=3e-11, rtol=0)
    assert np.allclose(ours_tiles.ground_resolution_many(geo[:, 0], level), [upstream_tiles.ground_resolution(lat, level) for lat in geo[:, 0]])


def test_mojo_batch_quadint_parity():
    rng = np.random.default_rng(3)
    level = 29
    tiles = rng.integers(0, 1 << level, size=(20_000, 2), dtype=np.int64)
    actual = ours_tiles.tile_to_quadint_many(tiles, level)
    expected = np.array([upstream_tiles.quadkey_to_quadint(upstream_tiles.tile_to_quadkey(tuple(row), level)) for row in tiles], dtype=np.uint64)
    assert np.array_equal(actual, expected)


def test_mojo_batch_boundary_validation_and_empty_inputs():
    empty_geo = np.empty((0, 2), dtype=np.float64)
    empty_pixel = np.empty((0, 2), dtype=np.int64)
    assert ours_tiles.geo_to_pixel_many(empty_geo, 18).shape == (0, 2)
    assert ours_tiles.pixel_to_geo_many(empty_pixel, 18).shape == (0, 2)
    assert ours_tiles.ground_resolution_many(np.array([], dtype=np.float64), 18).shape == (0,)
    assert ours_tiles.tile_to_quadint_many(empty_pixel, 18).shape == (0,)
    with pytest.raises(ValueError):
        ours_tiles.geo_to_pixel_many([[0.0, 0.0]], 30)
    with pytest.raises(TypeError):
        ours_tiles.pixel_to_geo_many([[1.5, 2.0]], 18)
    with pytest.raises(ValueError):
        ours_tiles.geo_to_pixel_many([[float("nan"), 0.0]], 18)
    with pytest.raises(OverflowError):
        ours_tiles.tile_to_quadint_many(np.array([[2**63, 0]], dtype=np.uint64), 18)
    with pytest.raises(ValueError):
        ours_tiles.ground_resolution_many([[0.0]], 18)
