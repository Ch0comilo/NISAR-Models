"""Exercise the embedded bootstrap against real offline Git clones and imports."""
from pathlib import Path
import importlib
import json
import runpy
import shutil
import subprocess
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/notebook_map_bootstrap.py"
load_map = runpy.run_path(str(BOOTSTRAP))["load_notebook_map"]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Requires local Git")


def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def write_map(project, version="esri-gibs-v1", real=False):
    package = project / "src/nisar_water"
    for name, content in {
        "notebook_aoi.py": 'MARKER = "current source"\n',
        "notebook_map.py": ('from .notebook_aoi import MARKER\n'
                            'class RectanglePicker:\n'
                            f'    basemap_version = "{version}"\n'),
    }.items():
        if real:
            content = (ROOT / "src/nisar_water" / name).read_text(encoding="utf-8")
        (package / name).write_text(content, encoding="utf-8")


@pytest.fixture
def isolated_package():
    saved = {name: module for name, module in sys.modules.copy().items()
             if name == "nisar_water" or name.startswith("nisar_water.")}
    old_path = sys.path[:]
    for name in saved:
        del sys.modules[name]
    yield
    for name in tuple(sys.modules):
        if name == "nisar_water" or name.startswith("nisar_water."):
            del sys.modules[name]
    sys.modules.update(saved)
    sys.path[:] = old_path
    importlib.invalidate_caches()


@pytest.fixture
def checkout(tmp_path, isolated_package):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    git(upstream, "config", "user.name", "Bootstrap test")
    git(upstream, "config", "user.email", "bootstrap@example.invalid")
    git(upstream, "config", "core.autocrlf", "false")
    project = upstream / "nisar-water"
    package = project / "src/nisar_water"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('VERSION = "before the map existed"\n', encoding="utf-8")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "Before notebook map")
    clone = tmp_path / "runtime"
    git(tmp_path, "clone", "--config", "core.autocrlf=false", str(upstream), str(clone))
    git(clone, "config", "core.autocrlf", "false")
    return upstream, clone / "nisar-water"


def publish_map(upstream, **kwargs):
    write_map(upstream / "nisar-water", **kwargs)
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "Add current notebook map")


def test_missing_module_is_updated_before_import_even_with_cached_package(checkout):
    upstream, project = checkout
    sys.path.insert(0, str(project / "src"))
    cached = importlib.import_module("nisar_water")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nisar_water.notebook_map")
    publish_map(upstream)

    module = load_map(project, str(upstream), allow_update=True)

    assert module.RectanglePicker.basemap_version == "esri-gibs-v1"
    assert module.MARKER == "current source"
    assert sys.modules["nisar_water"] is cached
    assert Path(module.__file__).resolve() == project / "src/nisar_water/notebook_map.py"
    assert git(project.parent, "rev-parse", "HEAD") == git(upstream, "rev-parse", "HEAD")


def test_stale_module_and_parent_attribute_are_reloaded(checkout):
    upstream, project = checkout
    publish_map(upstream, version="esri-gibs-v0")
    git(project.parent, "pull", "--ff-only", "origin", "main")
    sys.path.insert(0, str(project / "src"))
    stale = importlib.import_module("nisar_water.notebook_map")
    assert stale.RectanglePicker.basemap_version == "esri-gibs-v0"
    publish_map(upstream)

    current = load_map(project, str(upstream), allow_update=True)

    assert current is not stale
    assert current.RectanglePicker.basemap_version == "esri-gibs-v1"
    assert sys.modules["nisar_water"].notebook_map is current


def test_another_cached_clone_cannot_supply_the_map(checkout, tmp_path):
    upstream, project = checkout
    other = tmp_path / "other/src/nisar_water"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text('SOURCE = "other clone"\n', encoding="utf-8")
    (other / "legacy.py").write_text('SOURCE = "other clone"\n', encoding="utf-8")
    sys.path.insert(0, str(other.parent))
    stale_package = importlib.import_module("nisar_water")
    importlib.import_module("nisar_water.legacy")
    publish_map(upstream)

    current = load_map(project, str(upstream), allow_update=True)

    assert sys.modules["nisar_water"] is not stale_package
    assert "nisar_water.legacy" not in sys.modules
    assert sys.path[0] == str(project / "src")
    assert Path(current.__file__).resolve() == project / "src/nisar_water/notebook_map.py"


@pytest.mark.parametrize("reason", ["dirty", "other_remote", "local_session"])
def test_stale_checkout_is_not_changed_without_safe_update(checkout, reason):
    upstream, project = checkout
    publish_map(upstream)
    before = git(project.parent, "rev-parse", "HEAD")
    init = project / "src/nisar_water/__init__.py"
    if reason == "dirty":
        init.write_text('MY_LOCAL_WORK = "keep this"\n', encoding="utf-8")
    original = init.read_bytes()

    with pytest.raises(RuntimeError):
        load_map(project, "another repository" if reason == "other_remote" else str(upstream),
                 allow_update=reason != "local_session")

    assert git(project.parent, "rev-parse", "HEAD") == before
    assert init.read_bytes() == original
    assert not (project / "src/nisar_water/notebook_map.py").exists()


def test_current_sources_do_not_need_git_or_network(tmp_path, isolated_package, monkeypatch):
    project = tmp_path / "local project"
    package = project / "src/nisar_water"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    write_map(project)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected Git/network call"))

    assert load_map(project, "unused").MARKER == "current source"


def test_public_map_cell_is_self_contained_and_repairs_a_pre_map_clone(checkout, monkeypatch):
    pytest.importorskip("ipyleaflet")
    upstream, project = checkout
    publish_map(upstream, real=True)
    notebook = json.loads((ROOT / "notebooks/NISAR_Water_Try_Your_Data.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "roi_picker = RectanglePicker(" in "".join(cell["source"]))
    assert cell.startswith(BOOTSTRAP.read_text(encoding="utf-8"))
    colab = types.ModuleType("google.colab")
    enabled = []
    colab.output = types.SimpleNamespace(enable_custom_widget_manager=lambda: enabled.append(True))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {"PROJECT_ROOT": project, "REPOSITORY": str(upstream),
                 "INITIAL_BBOX": [104.18, 18.34, 104.26, 18.42]}

    exec(compile(cell, "public-notebook-map-cell", "exec"), namespace)

    picker = namespace["roi_picker"]
    try:
        assert enabled == [True]
        assert picker.selection()["bbox"] == namespace["INITIAL_BBOX"]
        assert picker.basemap_version == "esri-gibs-v1"
        assert "openstreetmap.org" not in picker.map.layers[0].url
    finally:
        picker.widget.close()
        picker.map.close()
        picker.draw.close()
