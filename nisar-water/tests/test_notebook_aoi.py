import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nisar_water.notebook_aoi import bbox_selection, rectangle_selection


BOUNDS = [-78.98765432101234, -3.123456789012345, -78.89987654321098, -3.018765432109876]


def test_asymmetric_southern_western_rectangle_preserves_exact_coordinates():
    result = bbox_selection(BOUNDS)
    assert result["bbox"] == BOUNDS
    west, south, east, north = BOUNDS
    assert result["aoi"]["coordinates"] == [[
        [west, south], [east, south], [east, north], [west, north], [west, south],
    ]]
    assert json.loads(json.dumps(result))["bbox"] == BOUNDS
    assert east-west != pytest.approx(north-south)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_drawn_rectangle_accepts_either_winding_and_any_start_corner(start, reverse):
    selection = bbox_selection(BOUNDS)
    points = selection["aoi"]["coordinates"][0][:-1]
    if reverse:
        points = list(reversed(points))
    points = points[start:] + points[:start]
    feature = {"type": "Feature", "properties": {"widget": "rectangle"},
               "geometry": {"type": "Polygon", "coordinates": [points + [points[0]]]}}
    original = copy.deepcopy(feature)
    assert rectangle_selection(feature) == selection
    assert feature == original
    assert rectangle_selection(feature["geometry"]) == selection


def test_edited_rectangle_uses_new_precise_bounds_without_old_values():
    edited = [-78.96543210987654, -3.102345678901234, -78.91234567890123, -3.034567890123456]
    assert rectangle_selection(bbox_selection(edited)["aoi"])["bbox"] == edited
    assert rectangle_selection(bbox_selection(edited)["aoi"])["bbox"] != BOUNDS


@pytest.mark.parametrize("bounds", [
    [0, 0, 0, .1], [0, 0, .1, 0], [.1, 0, 0, .1], [0, .1, .1, 0],
    [179.95, 0, -179.95, .1], [-180, 0, -179.9, .1], [179.9, 0, 180, .1],
    [0, -80.01, .1, -79.9], [0, 83.9, .1, 84.01],
    [0, 0, .201, .1], [0, 0, .1, .201],
    [0, 0, float("nan"), .1], [0, float("inf"), .1, .2],
    [True, 0, .1, .1], ["0", 0, .1, .1],
    [0, 0, .1], [0, 0, .1, .1, .1], None, "0,0,.1,.1",
])
def test_invalid_or_unsupported_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        bbox_selection(bounds)


def test_nominal_point_two_degree_boundary_is_not_rejected_by_float_subtraction():
    bounds = [-79.5, -3.2, -79.3, -3.0]
    assert bbox_selection(bounds)["bbox"] == bounds
    # This is only the coarse drawing guard; the worker also checks its halo.


@pytest.mark.parametrize("geometry", [
    None, {}, {"type": "Feature", "geometry": None},
    {"type": "FeatureCollection", "features": []},
    {"type": "Point", "coordinates": [0, 0]},
    {"type": "Polygon", "coordinates": []},
    {"type": "Polygon", "coordinates": [[[0, 0], [.1, 0], [.1, .1], [0, .1]]]},
    # Same rectangular envelope but a trapezoid must not become that envelope.
    {"type": "Polygon", "coordinates": [[[0, 0], [.1, 0], [.08, .1], [0, .1], [0, 0]]]},
    # A rotated square is not an axis-aligned drawing rectangle.
    {"type": "Polygon", "coordinates": [[[.05, 0], [.1, .05], [.05, .1], [0, .05], [.05, 0]]]},
    # All four corners in a self-intersecting order.
    {"type": "Polygon", "coordinates": [[[0, 0], [.1, .1], [0, .1], [.1, 0], [0, 0]]]},
    {"type": "Polygon", "coordinates": [[[0, 0], [.1, 0], [.1, .1], [0, .1], [0, .000001]]]},
    {"type": "Polygon", "coordinates": [[[0, 0], [.1, 0], [.1, .1], [0, .1], [0, 0]],
                                           [[.02, .02], [.03, .02], [.03, .03], [.02, .03], [.02, .02]]]},
    {"type": "Polygon", "coordinates": [[[0, 0, 20], [.1, 0, 20], [.1, .1, 20], [0, .1, 20], [0, 0, 20]]]},
])
def test_only_exact_simple_axis_aligned_rectangle_is_accepted(geometry):
    with pytest.raises(ValueError):
        rectangle_selection(geometry)


def test_helper_imports_and_runs_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    code = ("from nisar_water.notebook_aoi import bbox_selection; "
            "import sys; assert bbox_selection([-.1,-.2,0,-.1])['bbox']==[-.1,-.2,0.,-.1]; "
            "assert not {'numpy','pyproj','ipyleaflet'} & set(sys.modules)")
    subprocess.run([sys.executable, "-S", "-c", code], env=environment,
                   check=True, capture_output=True, text=True)
