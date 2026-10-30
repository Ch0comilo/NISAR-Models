"""Stdlib-only bootstrap embedded in the public notebook, including in old clones."""
from pathlib import Path
import ast
import importlib
import importlib.util
import subprocess
import sys


def load_notebook_map(project_root, repository, *, allow_update=False):
    """Check source files before importing; refresh only a clean matching clone.

    This code is embedded in the notebook rather than imported from the checkout:
    a Colab runtime may still have a checkout from before notebook_map existed.
    The model runs in its own interpreter; only the kernel's map imports change.
    """
    project_root = Path(project_root).resolve()
    package_root = project_root / "src/nisar_water"

    def source_is_current():
        if not all((package_root / name).is_file() for name in (
            "__init__.py", "notebook_aoi.py", "notebook_map.py"
        )):
            return False
        try:
            tree = ast.parse((package_root / "notebook_map.py").read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return False
        return any(
            isinstance(node, ast.ClassDef) and node.name == "RectanglePicker"
            and any(isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "basemap_version"
                            for target in statement.targets)
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value == "esri-gibs-v1"
                    for statement in node.body)
            for node in tree.body
        )

    if not source_is_current():
        if not allow_update:
            raise RuntimeError("Falta el módulo actualizado del mapa. Actualiza el repositorio "
                               "con git pull --ff-only y vuelve a ejecutar esta celda.")
        repo_root = project_root.parent

        def git_output(*arguments):
            return subprocess.run(["git", "-C", str(repo_root), *arguments], check=True,
                                  capture_output=True, text=True).stdout.strip()

        remote = git_output("remote", "get-url", "origin")
        dirty = git_output("status", "--porcelain", "--untracked-files=no")
        if remote != repository or dirty:
            raise RuntimeError("La copia del mapa es antigua y contiene cambios locales "
                               "o usa otro repositorio. Actualízala manualmente conservando "
                               "tus cambios, o abre una sesión nueva de Colab.")
        print("Actualizando los archivos del mapa en la copia de Colab...")
        subprocess.run(["git", "-C", str(repo_root), "pull", "--ff-only", "origin", "main"],
                       check=True)
        if not source_is_current():
            raise RuntimeError("El repositorio actualizado aún no contiene el módulo "
                               "del mapa esperado. Abre la versión actual del notebook.")

    # Prepend even when this source directory was already later in sys.path.
    source = str(package_root.parent)
    sys.path[:] = [source, *(entry for entry in sys.path if entry != source)]
    cached = sys.modules.get("nisar_water")
    if cached is not None:
        locations = {Path(entry).resolve() for entry in getattr(cached, "__path__", ())}
        if locations != {package_root}:
            # An earlier pip install or another clone may own the cached namespace.
            for name in tuple(sys.modules):
                if name == "nisar_water" or name.startswith("nisar_water."):
                    del sys.modules[name]
            cached = None
    for name in ("notebook_map", "notebook_aoi"):
        sys.modules.pop("nisar_water." + name, None)
        if cached is not None:
            cached.__dict__.pop(name, None)
        # A same-size Git update within one second can leave timestamp-based
        # bytecode apparently valid. Recompile these two UI modules from source.
        bytecode = Path(importlib.util.cache_from_source(str(package_root / (name + ".py"))))
        bytecode.unlink(missing_ok=True)
    importlib.invalidate_caches()
    return importlib.import_module("nisar_water.notebook_map")
