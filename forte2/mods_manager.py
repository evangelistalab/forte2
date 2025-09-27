import importlib
import pkgutil
import os
import sys
from pathlib import Path


def load_mods():
    """
    Load forte2 mods installed from one or more directories.

    By default, this function checks for mods in the following directories:
    1. The ~/.forte2/mods/ directory in the user's home folder.
    2. The mods/ directory in the forte2 package.

    Mods are expected to be python files with a register(forte2) function of the form

    ```python
    # ~/.forte2/mods/my_mod.py

    def register(forte2):
        ...
    ```
    """
    user_path = Path.home() / ".forte2/mods"
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


def enable_mod(modname: str, paths=None):
    """
    Load a specific forte2 mod

    By default, this function checks for mods in the following directories:
    1. The ~/.forte2/mods/ directory in the user's home folder.
    2. The mods/ directory in the forte2 package.
    3. The current working directory.

    Mods are expected to be python files with a register(forte2) function of the form

    ```python
    # ~/.forte2/mods/my_mod.py

    def register(forte2):
        ...
    ```

    Parameters
    ----------
    modname : str
        The name of the mod to load (without the .py extension)
    paths : list of Path or str, optional, default=None
        Optional list of directories to search for mods.
        If None, the default directories are used.
    """
    if paths is None:
        user_path = Path.home() / ".forte2/optional_mods"
        package_path = Path(__file__).resolve().parent.parent / "optional_mods"
        # include the folder in which the script is executed
        if os.getcwd() != str(user_path) and os.getcwd() != str(package_path):
            paths = [Path(os.getcwd()), user_path, package_path]
        else:
            paths = [user_path, package_path]

    for path in paths:
        if not path.exists():
            continue

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        for _, name, _ in pkgutil.iter_modules([str(path)]):
            if name != modname:
                continue
            try:
                print(f"[mods_manager] loading mod {name} from {path}")
                mod = importlib.import_module(name)
                if hasattr(mod, "register"):
                    mod.register(importlib.import_module("forte2"))
            except Exception as e:
                print(f"[mods_manager] failed to load mod {name} from {path}: {e}")
