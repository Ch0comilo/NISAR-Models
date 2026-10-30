"""The widget's synchronized drawing state is the single source of the AOI."""
import json

import pytest

pytest.importorskip("ipyleaflet")
from nisar_water.notebook_aoi import bbox_selection
from nisar_water.notebook_map import RectanglePicker


DEFAULT = [104.18, 18.34, 104.26, 18.42]


def feature(bbox):
    return {"type": "Feature", "properties": {}, "geometry": bbox_selection(bbox)["aoi"]}


@pytest.fixture
def picker():
    picker = RectanglePicker(DEFAULT)
    yield picker
    picker.widget.close()
    picker.map.close()
    picker.draw.close()


def test_browser_create_and_edit_data_replace_coordinates_without_rounding(picker):
    drawn = [-75.123456789012, -6.987654321012, -75.054321098765, -6.912345678901]
    picker.draw.data = [feature(DEFAULT), feature(drawn)]
    assert len(picker.draw.data) == 1
    assert picker.selection() == bbox_selection(drawn)
    assert json.loads(picker.coordinates.value) == drawn
    edited = [-75.11987654321, -6.98, -75.06, -6.92]
    picker.draw.data = [feature(edited)]
    assert picker.selection() == bbox_selection(edited)


def test_deleted_rectangle_blocks_previous_snapshot_and_can_be_restored(picker):
    previous = picker.selection()
    picker.draw.data = []
    assert picker.coordinates.value == ""
    with pytest.raises(ValueError, match="rectángulo"):
        picker.selection()
    picker.reset.click()
    assert picker.selection() == previous


def test_invalid_new_drawing_cannot_fall_back_to_the_previous_default(picker):
    invalid = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
        [[-75, -6], [-74, -6], [-74, -5], [-75, -5], [-75, -6]]]}, "properties": {}}
    picker.draw.data = [feature(DEFAULT), invalid]
    assert len(picker.draw.data) == 1
    with pytest.raises(ValueError, match="rectángulo"):
        picker.selection()


def test_manual_coordinates_require_apply_and_then_move_the_map(picker):
    new = [-74.1, 4.5, -74.05, 4.55]
    picker.coordinates.value = json.dumps(new)
    with pytest.raises(ValueError, match="sin aplicar"):
        picker.selection()
    picker.apply.click()
    assert picker.selection() == bbox_selection(new)
    assert list(picker.map.center) == [(new[1] + new[3]) / 2, (new[0] + new[2]) / 2]


def test_invalid_manual_coordinates_block_inference_and_drawing_recovers(picker):
    picker.coordinates.value = "[NaN, 4.5, -74.05, 4.55]"
    picker.apply.click()
    with pytest.raises(ValueError):
        picker.selection()
    picker.draw.data = [feature([-74.1, 4.5, -74.05, 4.55])]
    assert picker.selection()["bbox"][0] == -74.1


def test_moving_the_viewport_does_not_replace_the_selected_rectangle(picker):
    expected = picker.selection()
    picker.map.center = (4.7, -74.1)
    picker.map.zoom = 7
    assert picker.selection() == expected


def test_selection_snapshot_does_not_mutate_after_later_edit(picker):
    snapshot = picker.selection()
    snapshot["bbox"][0] = 0
    assert picker.selection()["bbox"] == DEFAULT
    picker.clear.click()
    with pytest.raises(ValueError):
        picker.selection()
