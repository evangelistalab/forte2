import shutil
import tempfile

import pytest

try:
    from forte2.dmrg import BLOCK2_AVAILABLE
except ImportError:
    BLOCK2_AVAILABLE = False

requires_block2 = pytest.mark.skipif(
    not BLOCK2_AVAILABLE,
    reason="block2 (pyblock2) is not installed.",
)


def _detect_block2_complex():
    """
    The relativistic DMRG solver needs block2 built with complex + general-spin
    support (SymmetryTypes.SGF | SymmetryTypes.CPX). The PyPI wheels ship it, but
    a source build may omit it, so detect it explicitly.
    """
    if not BLOCK2_AVAILABLE:
        return False
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        scratch = tempfile.mkdtemp(prefix="forte2_b2cpx_probe_")
        try:
            DMRGDriver(
                scratch=scratch,
                symm_type=SymmetryTypes.SGF | SymmetryTypes.CPX,
                n_threads=1,
            )
            return True
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception:
        return False


BLOCK2_COMPLEX_AVAILABLE = _detect_block2_complex()

requires_block2_complex = pytest.mark.skipif(
    not BLOCK2_COMPLEX_AVAILABLE,
    reason="block2 is not built with complex + general-spin (SGF|CPX) support.",
)
