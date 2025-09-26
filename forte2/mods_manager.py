import importlib
import pkgutil
import os
import sys
from pathlib import Path


def load_mods(paths=None):
    """
    Load mods from one or more directories.
    By default: first ~/.forte2, then mods/ shipped with the package.
    User mods can override shipped mods.
    """
    if paths is None:
        user_path = Path.home() / ".forte2"
        package_path = Path(__file__).resolve().parent.parent / "mods"
        paths = [user_path, package_path]

    for path in paths:
        if not path.exists():
            continue

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        for _, name, _ in pkgutil.iter_modules([str(path)]):
            try:
                print(f"[mods_manager] loading mod {name} from {path}")
                mod = importlib.import_module(name)
                if hasattr(mod, "register"):
                    mod.register(importlib.import_module("forte2"))
            except Exception as e:
                print(f"[mods_manager] failed to load mod {name} from {path}: {e}")
