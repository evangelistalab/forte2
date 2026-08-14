import os
import fnmatch
import __main__

# Shell-style glob patterns (``*``, ``?``, ``[seq]``)
ECHO_BLACKLIST = ["test_*"]


def _blacklisted(script_path):
    """True if the file name of `script_path` matches an ECHO_BLACKLIST pattern."""
    name = os.path.basename(script_path)
    return any(fnmatch.fnmatch(name, pattern) for pattern in ECHO_BLACKLIST)


def _do_not_echo(script_path):
    """
    True under IPython, a Jupyter kernel, pytest, or a file matching the blacklist
    False for a plain script.
    """
    try:
        get_ipython()  # noqa: F821 -- injected into builtins by IPython/Jupyter
        return True
    except NameError:
        pass
    # Set by pytest itself for the whole session; unlike `"pytest" in sys.modules`,
    # this isn't a false positive when forte2 merely imports pytest (e.g. for
    # pytest.approx in helpers.comparisons).
    if "PYTEST_VERSION" in os.environ:
        return True
    return _blacklisted(script_path)


def echo_invoking_script():
    """Print the source of the top-level script to stdout.

    Runs only for a plain command-line invocation (``python script.py`` or
    ``python -m ...``).
    Does not run under IPython, Jupyter, an interactive
    REPL, or pytest, where the script content isn't useful to echo, nor for
    scripts whose file name matches ``ECHO_BLACKLIST``.
    """
    script_path = getattr(__main__, "__file__", None)
    if script_path is None:
        return

    if _do_not_echo(script_path):
        return

    try:
        with open(script_path) as f:
            content = f.read()
    except OSError:
        return

    banner = f"Content of {script_path}"
    rule = "=" * len(banner)
    print(f"{rule}\n{banner}\n{rule}")
    print(content)
    print(rule)
