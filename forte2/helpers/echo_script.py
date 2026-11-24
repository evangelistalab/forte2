import atexit
import os
import fnmatch
import socket
import sys
import time
from datetime import datetime

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


def _plain_script_path():
    """Path of the top-level script for a plain command-line run, else None."""
    script_path = getattr(__main__, "__file__", None)
    if script_path is None or _do_not_echo(script_path):
        return None
    return script_path


def echo_invoking_script():
    """Print the source of the top-level script to stdout.

    Runs only for a plain command-line invocation (``python script.py`` or
    ``python -m ...``).
    Does not run under IPython, Jupyter, an interactive
    REPL, or pytest, where the script content isn't useful to echo, nor for
    scripts whose file name matches ``ECHO_BLACKLIST``.
    """
    script_path = _plain_script_path()
    if script_path is None:
        return

    try:
        with open(script_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        print(f"Could not echo the invoking script {script_path}: {e}")
        return

    banner = f"Content of {script_path}"
    rule = "=" * len(banner)
    print(f"{rule}\n{banner}\n{rule}")
    print(content)
    print(rule)


def _timestamp():
    """Current local time with its time zone, e.g. 2026-11-17 15:30:12 EST."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_duration(seconds):
    """Format a duration as 3.2s, 12m 45s, or 1h 38m 2s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def report_run_timing():
    """
    Print when and where the run started, and at exit when it ended and how long it took.
    """
    if _plain_script_path() is None:
        return

    print(f"forte2 run started  {_timestamp()} on {socket.gethostname()}")
    wall_start = time.monotonic()
    cpu_start = time.process_time()
    failure = []

    previous_excepthook = sys.excepthook

    def excepthook(exc_type, exc, tb):
        failure.append(exc_type.__name__)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = excepthook

    def report_end():
        wall = _format_duration(time.monotonic() - wall_start)
        cpu = _format_duration(time.process_time() - cpu_start)
        if failure:
            status = f"failed   {_timestamp()} with {failure[0]}"
        else:
            status = f"finished {_timestamp()}"
        print(f"forte2 run {status} (wall {wall}, CPU {cpu})", flush=True)

    atexit.register(report_end)
