"""Expand the composite-index DSRG-MRPT3 terms into elementary block contractions."""

import itertools
import sys

sys.path.insert(0, "codegen")
from terms import TERMS, LETTER, EXPAND

OD1 = ("ac", "vc", "va", "ca", "cv", "av")
# pphh and hhpp, less the all-active block they share: that is an internal
# excitation, which the amplitudes never carry.
OD2 = tuple(
    sorted(
        (
            {"".join(t) for t in itertools.product("av", "av", "ca", "ca")}
            | {"".join(t) for t in itertools.product("ca", "ca", "av", "av")}
        )
        - {"aaaa"}
    )
)
ALL1 = tuple("".join(t) for t in itertools.product("cav", repeat=2))
T1_SUPPORT = ("ca", "cv", "av")
T2_SUPPORT = tuple(
    sorted({"".join(t) for t in itertools.product("ca", "ca", "av", "av")} - {"aaaa"})
)
DENSITY = {"G": "g1", "E": "e1", "L": "l2"}


def expand(spec, kinds, pair, h2_od):
    """Yield (out_block, pair_block, operand_blocks, needs_df) for one term.

    `h2_od` prunes the two-body operand to off-diagonal blocks, which is what it
    is when the operand is a commutator rather than the bare Hamiltonian.
    """
    ins, res = spec.split("->")
    ins = ins.split(",")
    letters = sorted(set("".join(ins)))
    ref = OD1 if len(res) == 2 else OD2
    for assign in itertools.product(*[EXPAND[LETTER[c]] for c in letters]):
        amap = dict(zip(letters, assign))
        out = "".join(amap[c] for c in res)
        if out not in ref:
            continue
        blocks, ok, df = [], True, False
        for idx, kind in zip(ins, kinds):
            blk = "".join(amap[c] for c in idx)
            if kind == "T1" and blk not in T1_SUPPORT:
                ok = False
                break
            if kind in ("T2", "S2") and blk not in T2_SUPPORT:
                ok = False
                break
            if kind == "H2":
                if h2_od and blk not in OD2:
                    ok = False
                    break
                if blk.count("v") >= 3:
                    df = True
            blocks.append(blk)
        if not ok:
            continue
        if pair:
            perm = (1, 0) if len(res) == 2 else (1, 0, 3, 2)
            pblk = "".join(out[i] for i in perm)
        else:
            pblk = None
        yield out, pblk, tuple(blocks), df


if __name__ == "__main__":
    for od in (False, True):
        n = d = 0
        for kern, terms in TERMS.items():
            for coef, spec, kinds, pair in terms:
                for _, _, _, df in expand(spec, kinds, pair, od):
                    n += 1
                    d += df
        tag = "off-diagonal operand" if od else "bare Hamiltonian operand"
        print(f"RESULT {tag:26s}: {n:4d} contractions, {d:3d} need DF")
