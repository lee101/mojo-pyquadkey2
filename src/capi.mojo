"""Batch Web-Mercator and quadint kernels exposed through a stable C ABI."""

from std.math import atan, cos, exp, log, sin


comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IntPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def f64_ptr(address: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=address)


def int_ptr(address: Int) -> IntPtr:
    return IntPtr(unsafe_from_address=address)


def clipped(value: Float64, low: Float64, high: Float64) -> Float64:
    return min(max(value, low), high)


@export("mpq_geo_to_pixel")
def mpq_geo_to_pixel(
    latitude: Int, longitude: Int, pixel_x: Int, pixel_y: Int, n: Int, level: Int
) abi("C"):
    if n <= 0 or level < 1 or level > 29 or latitude == 0 or longitude == 0 or pixel_x == 0 or pixel_y == 0:
        return
    var latitudes = f64_ptr(latitude)
    var longitudes = f64_ptr(longitude)
    var xs = int_ptr(pixel_x)
    var ys = int_ptr(pixel_y)
    var map_size = Float64(256 * (1 << level))
    var pi = 3.14159265358979323846264338327950288
    for i in range(n):
        var lat = clipped(latitudes[i], -85.05112878, 85.05112878)
        var lon = clipped(longitudes[i], -180.0, 180.0)
        var x = (lon + 180.0) / 360.0
        var sin_lat = sin(lat * pi / 180.0)
        var y = 0.5 - log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * pi)
        xs[i] = Int(clipped(x * map_size + 0.5, 0.0, map_size - 1.0))
        ys[i] = Int(clipped(y * map_size + 0.5, 0.0, map_size - 1.0))


@export("mpq_pixel_to_geo")
def mpq_pixel_to_geo(
    pixel_x: Int, pixel_y: Int, latitude: Int, longitude: Int, n: Int, level: Int
) abi("C"):
    if n <= 0 or level < 1 or level > 29 or pixel_x == 0 or pixel_y == 0 or latitude == 0 or longitude == 0:
        return
    var xs = int_ptr(pixel_x)
    var ys = int_ptr(pixel_y)
    var latitudes = f64_ptr(latitude)
    var longitudes = f64_ptr(longitude)
    var map_size = Float64(256 * (1 << level))
    var pi = 3.14159265358979323846264338327950288
    for i in range(n):
        var x = clipped(Float64(xs[i]), 0.0, map_size - 1.0) / map_size - 0.5
        var y = 0.5 - clipped(Float64(ys[i]), 0.0, map_size - 1.0) / map_size
        var lat = 90.0 - 360.0 * atan(exp(-y * 2.0 * pi)) / pi
        latitudes[i] = Float64(Int(lat * 1000000000000.0)) / 1000000000000.0
        longitudes[i] = Float64(Int(360.0 * x * 1000000000000.0)) / 1000000000000.0


@export("mpq_ground_resolution")
def mpq_ground_resolution(latitude: Int, result: Int, n: Int, level: Int) abi("C"):
    if n <= 0 or level < 1 or level > 29 or latitude == 0 or result == 0:
        return
    var latitudes = f64_ptr(latitude)
    var resolutions = f64_ptr(result)
    var map_size = Float64(256 * (1 << level))
    var pi = 3.14159265358979323846264338327950288
    for i in range(n):
        var lat = clipped(latitudes[i], -85.05112878, 85.05112878)
        resolutions[i] = cos(lat * pi / 180.0) * 2.0 * pi * 6378137.0 / map_size


@export("mpq_tile_to_quadint")
def mpq_tile_to_quadint(tile_x: Int, tile_y: Int, result: Int, n: Int, level: Int) abi("C"):
    if n <= 0 or level < 1 or level > 29 or tile_x == 0 or tile_y == 0 or result == 0:
        return
    var xs = int_ptr(tile_x)
    var ys = int_ptr(tile_y)
    var quadints = int_ptr(result)
    for row in range(n):
        var value = level
        for i in range(level):
            var bit_location = 64 - (i + 1) * 2
            var mask = 1 << (level - i - 1)
            var digit = 0
            if (xs[row] & mask) != 0:
                digit += 1
            if (ys[row] & mask) != 0:
                digit += 2
            value |= digit << bit_location
        quadints[row] = value
