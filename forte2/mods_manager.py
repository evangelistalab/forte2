import importlib
import pkgutil
import os
import sys
from pathlib import Path

from forte2.helpers import logger


def load_mods():
    """
    Load forte2 installed mods from standard locations.

    By default, this function checks for mods in the following directories:
    1. The ~/.forte2/mods/ directory in the user's home folder.
    2. The mods/ directory in the forte2 package.

    Mods are expected to be python files with a register(forte2) function of the form

    ```python
    # ~/.forte2/mods/my_mod/my_mod.py

    def new_feature(**args, **kwargs):
        ...

    def register(forte2):
        forte2.new_feature = new_feature
    ```

    And remember to include an __init__.py file in the mods/my_mod/ directory to make it a package:

    ```python
    # ~/.forte2/mods/my_mod/__init__.py
    from .my_mod import register
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
                logger.log(f"[mods_manager] loading mod {name} from {path}")
                mod = importlib.import_module(name)
                if hasattr(mod, "register"):
                    mod.register(importlib.import_module("forte2"))
                else:
                    logger.log_warning(
                        f"[mods_manager] mod {name} from {path} does not have a register(forte2) function"
                    )
            except Exception as e:
                logger.log_warning(
                    f"[mods_manager] failed to load mod {name} from {path}: {e}"
                )


def enable_mod(modname: str, paths=None):
    """
    Load optional mods

    By default, this function checks for mods in the following directories:
    1. The ~/.forte2/optional_mods/ directory in the user's home folder.
    2. The current working directory.

    Optional mods are expected to be python files with a register(forte2) function of the form

    ```python
    # ~/.forte2/optional_mods/my_mod/my_mod.py

    def new_feature(**args, **kwargs):
        ...

    def register(forte2):
        forte2.new_feature = new_feature
    ```

    And remember to include an __init__.py file in the optional_mods/my_mod/ directory to make it a package:

    ```python
    # ~/.forte2/optional_mods/my_mod/__init__.py
    from .my_mod import register
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
        # include the folder in which the script is executed only if it's not the same as user_path
        if os.getcwd() != str(user_path):
            paths = [Path(os.getcwd()), user_path]
        else:
            paths = [user_path]

    for path in paths:
        if not path.exists():
            continue

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        for _, name, _ in pkgutil.iter_modules([str(path)]):
            if name != modname:
                continue
            try:
                logger.log(f"[mods_manager] loading mod {name} from {path}")
                mod = importlib.import_module(name)
                if hasattr(mod, "register"):
                    mod.register(importlib.import_module("forte2"))
                    # mod found and loaded, return
                    return
            except Exception as e:
                logger.log_warning(
                    f"[mods_manager] failed to load mod {name} from {path}: {e}"
                )
