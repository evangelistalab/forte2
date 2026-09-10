import itertools

import numpy as np
import pytest

from forte2.dsrg.dsrg_common import _DSRGDenseHelper
from forte2.dsrg.dsrg_mrpt3_kernels import (
    ALL1_LABELS,
    OD2_LABELS,
    T1_LABELS,
    T2_LABELS,
    _DSRGBlockHelper,
)

# Small enough to be quick, but with every space a different size: equal
# dimensions would let an index-ordering mistake pass unnoticed.
NC, NA, NV = 2, 3, 4
NCORR, NHOLE, NPART = NC + NA + NV, NC + NA, NA + NV
NAUX = 5

OD1_LABELS = ("ac", "vc", "va", "ca", "cv", "av")


class _Spaces:
    """The dimensions and slices a helper reads off the DSRG object."""

    ncore, nact, nvirt = NC, NA, NV
    ncorr, nhole, npart = NCORR, NHOLE, NPART
    core, actv, virt = slice(0, NC), slice(NC, NC + NA), slice(NC + NA, NCORR)
    hole, part = slice(0, NHOLE), slice(NC, NCORR)
    hc, ha = slice(0, NC), slice(NC, NHOLE)
    pa, pv = slice(0, NA), slice(NA, NPART)


CORR = {"c": _Spaces.core, "a": _Spaces.actv, "v": _Spaces.virt}
HOLE = {"c": slice(0, NC), "a": slice(NC, NHOLE)}
PART = {"a": slice(0, NA), "v": slice(NA, NPART)}


def _is_ph(label):
    """Whether a one-body block runs particle then hole.

    Both halves must be checked: the active space belongs to both, so "av"
    starts with a particle-space letter yet is hole-particle.
    """
    return label[0] in "av" and label[1] in "ca"


def _is_pphh(label):
    """The two-body counterpart; "ccaa" ends in two active indices yet is hhpp."""
    return all(c in "av" for c in label[:2]) and all(c in "ca" for c in label[2:])


@pytest.fixture(scope="module")
def operands():
    """The same random operands in dense and in block form.

    The integrals are built from three-index factors rather than drawn at
    random, so the blocks the kernels rebuild that way are exact rather than
    approximate.
    """
    rng = np.random.default_rng(7)
    B = rng.random((NAUX, NCORR, NCORR))
    Vd = np.einsum("Qpr,Qqs->pqrs", B, B)
    Fd = rng.random((NCORR, NCORR))

    T1d = rng.random((NHOLE, NPART))
    T1d[_Spaces.ha, _Spaces.pa] = 0.0
    T2d = rng.random((NHOLE, NHOLE, NPART, NPART))
    T2d[_Spaces.ha, _Spaces.ha, _Spaces.pa, _Spaces.pa] = 0.0
    S2d = 2 * T2d - T2d.swapaxes(2, 3)

    # a commutator has only off-diagonal blocks
    Cp = rng.random((NPART, NPART, NHOLE, NHOLE))
    Ch = rng.random((NHOLE, NHOLE, NPART, NPART))
    Cp[_Spaces.pa, _Spaces.pa, _Spaces.ha, _Spaces.ha] = 0.0
    Ch[_Spaces.ha, _Spaces.ha, _Spaces.pa, _Spaces.pa] = 0.0

    cum = {
        "gamma1": rng.random((NA, NA)),
        "eta1": rng.random((NA, NA)),
        "lambda2": rng.random((NA, NA, NA, NA)),
    }
    blocks = {
        "V": {
            b: Vd[tuple(CORR[c] for c in b)]
            for b in ("".join(t) for t in itertools.product("cav", repeat=4))
        },
        "F": {b: Fd[CORR[b[0]], CORR[b[1]]] for b in ALL1_LABELS},
        "B": {
            b: np.ascontiguousarray(B[:, CORR[b[0]], CORR[b[1]]])
            for b in ("".join(t) for t in itertools.product("cav", repeat=2))
        },
        "T1": {b: T1d[HOLE[b[0]], PART[b[1]]] for b in T1_LABELS},
        "T2": {
            b: T2d[HOLE[b[0]], HOLE[b[1]], PART[b[2]], PART[b[3]]] for b in T2_LABELS
        },
        "S2": {
            b: S2d[HOLE[b[0]], HOLE[b[1]], PART[b[2]], PART[b[3]]] for b in T2_LABELS
        },
        "C": {
            b: (
                Cp[PART[b[0]], PART[b[1]], HOLE[b[2]], HOLE[b[3]]]
                if _is_pphh(b)
                else Ch[HOLE[b[0]], HOLE[b[1]], PART[b[2]], PART[b[3]]]
            )
            for b in OD2_LABELS
        },
    }
    dense = {"V": Vd, "F": Fd, "T1": T1d, "T2": T2d, "S2": S2d, "C": (Cp, Ch)}
    return dense, blocks, cum


def _as_blocks(out, ndim):
    """Read the dense helper's two direction arrays as elementary blocks."""
    p, h = out
    if ndim == 2:
        return {
            b: (p[PART[b[0]], HOLE[b[1]]] if _is_ph(b) else h[HOLE[b[0]], PART[b[1]]])
            for b in OD1_LABELS
        }
    return {
        b: (
            p[PART[b[0]], PART[b[1]], HOLE[b[2]], HOLE[b[3]]]
            if _is_pphh(b)
            else h[HOLE[b[0]], HOLE[b[1]], PART[b[2]], PART[b[3]]]
        )
        for b in OD2_LABELS
    }


# (kernel, output rank, operand keys, whether the operand is itself a commutator)
CASES = [
    ("H1_T1_C1", 2, ("F", "T1"), False),
    ("H1_T2_C1", 2, ("F", "T2"), False),
    ("H1_T2_C2", 4, ("F", "T2"), False),
    ("H2_T1_C1", 2, ("V", "T1"), False),
    ("H2_T1_C2", 4, ("V", "T1"), False),
    ("H2_T2_C1", 2, ("V", "T2", "S2"), False),
    ("H2_T2_C2", 4, ("V", "T2", "S2"), False),
    ("H2_T1_C1", 2, ("C", "T1"), True),
    ("H2_T1_C2", 4, ("C", "T1"), True),
    ("H2_T2_C1", 2, ("C", "T2", "S2"), True),
    ("H2_T2_C2", 4, ("C", "T2", "S2"), True),
]


@pytest.mark.parametrize("name,ndim,keys,od", CASES)
def test_block_kernels_match_dense(operands, name, ndim, keys, od):
    """The generated block kernels must reproduce the dense reference exactly.

    `_DSRGDenseHelper` holds the same working equations in composite indices and
    is kept as ground truth, so a failure here names the kernel it is in. The
    energy tests would only show a shifted total.
    """
    dense_ops, block_ops, cum = operands
    dense, block = _DSRGDenseHelper(_Spaces()), _DSRGBlockHelper(_Spaces())
    dense.set_cumulants(cum)

    ref_out = dense.make_1body() if ndim == 2 else dense.make_2body()
    getattr(dense, name)(ref_out, *(dense_ops[k] for k in keys), 1.0)
    ref = _as_blocks(ref_out, ndim)

    got = block.make_1body() if ndim == 2 else block.make_2body()
    fn = getattr(block, name + ("_od" if od else ""))
    args = (got, *(block_ops[k] for k in keys), cum, 1.0)
    try:
        fn(*args, B=block_ops["B"])
    except TypeError:
        # a kernel with no three-virtual term takes no three-index integrals
        fn(*args)

    for b in ref:
        if b in got:
            assert np.max(np.abs(ref[b] - got[b])) < 1e-10, f"{name} block {b}"


def test_internal_blocks_are_absent():
    """All-active blocks are internal excitations and must not be storable.

    Making them structurally absent is stronger than zeroing them after the
    fact: nothing can write one back in.
    """
    assert "aa" not in T1_LABELS
    assert "aaaa" not in T2_LABELS
    assert "aaaa" not in OD2_LABELS
