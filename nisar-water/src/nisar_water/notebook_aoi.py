"""Exact rectangle selection for notebook kernels without geospatial packages.

Coordinates are WGS84 longitude/latitude. This module validates the drawing's
geometry and coarse bounds only; the scientific worker separately validates
its projected area, pixel budget and processing halo. No coordinates are
rounded, expanded, snapped to a grid or replaced with a polygon's envelope.
"""
from collections.abc import Mapping, Sequence
import math
from numbers import Real


def _sequence(value, length, description):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(description)
    return value


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Rectangle coordinates must be finite real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Rectangle coordinates must be finite real numbers")
    return result


def bbox_selection(bbox):
    """Return an exact rectangle from ordered [west, south, east, north].

    Only the current supported nonpolar, non-antimeridian profile is accepted.
    The 0.2-degree comparison allows floating-point subtraction roundoff; it
    never changes the coordinate values. Worker halo limits can be stricter.
    """
    values = _sequence(bbox, 4, "Use [west, south, east, north]")
    west, south, east, north = [_coordinate(value) for value in values]
    if not (-180 < west < east < 180 and -80 <= south < north <= 84):
        raise ValueError("Use ordered WGS84 bounds outside polar/antimeridian regions")
    for extent in (east-west, north-south):
        if extent > .2 and not math.isclose(extent, .2, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Rectangle must be at most 0.2 degrees per side before the processing halo")
    return {"bbox": [west, south, east, north], "aoi": {
        "type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south],
        ]],
    }}


def rectangle_selection(geo_json):
    """Validate a drawn Feature/Polygon as a single axis-aligned rectangle.

    Accept any starting corner and either winding order. The result uses the
    same four exact floating-point corner coordinates in canonical order.
    Holes, open rings, rotated rectangles and arbitrary polygons are rejected.
    """
    if not isinstance(geo_json, Mapping):
        raise ValueError("Draw a rectangle as a GeoJSON Polygon or Feature")
    geometry = geo_json.get("geometry") if geo_json.get("type") == "Feature" else geo_json
    if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
        raise ValueError("Draw one GeoJSON Polygon rectangle")
    rings = _sequence(geometry.get("coordinates"), 1, "A rectangle has one ring and no holes")
    ring = _sequence(rings[0], 5, "A rectangle needs four corners and an exact closing point")
    points = [tuple(_coordinate(value) for value in _sequence(point, 2, "Use longitude/latitude coordinate pairs"))
              for point in ring]
    if points[0] != points[-1]:
        raise ValueError("The rectangle ring must close exactly")
    corners = points[:-1]
    west, east = min(p[0] for p in corners), max(p[0] for p in corners)
    south, north = min(p[1] for p in corners), max(p[1] for p in corners)
    expected = {(west, south), (east, south), (east, north), (west, north)}
    if len(expected) != 4 or set(corners) != expected:
        raise ValueError("Draw an axis-aligned rectangle, not an arbitrary polygon")
    for first, second in zip(points, points[1:]):
        if first == second or ((first[0] == second[0]) == (first[1] == second[1])):
            raise ValueError("Rectangle edges must follow longitude or latitude exactly")
    return bbox_selection([west, south, east, north])
