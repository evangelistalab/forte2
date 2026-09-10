"""Generate the elementary-block DSRG-MRPT3 contraction kernels.

The working equations are held once, in composite indices, in codegen/terms.py.
This expands them over the elementary core/active/virtual blocks, prunes the
combinations that cannot contribute, and writes the result as explicit einsum
calls. Regenerate with:  python codegen/gen_kernels.py > forte2/dsrg/dsrg_mrpt3_kernels.py
"""

import sys

sys.path.insert(0, "codegen")

OUT = "forte2/dsrg/dsrg_mrpt3_kernels.py"
from terms import TERMS
from expand import expand, OD1, OD2, ALL1, T2_SUPPORT, T1_SUPPORT, DENSITY

OPERAND = {
    "H1": 'F["{}"]',
    "H2": 'V["{}"]',
    "T1": 'T1["{}"]',
    "T2": 'T2["{}"]',
    "S2": 'S2["{}"]',
    "G": "g1",
    "E": "e1",
    "L": "l2",
}
HAS_H2 = {k for k, ts in TERMS.items() if any("H2" in kinds for _, _, kinds, _ in ts)}


def coef_src(c):
    """The repo writes coefficients explicit-sign and three-decimal."""
    return f"scale * {c:+.3f}"


def emit_term(coef, spec, kinds, pair, h2_od, out_name):
    """Lines for one composite term, expanded over blocks."""
    lines, dfs = [], []
    rows = list(expand(spec, kinds, pair, h2_od))
    if not rows:
        return lines, dfs
    lines.append(f"        # {coef:+g} * {spec}")
    for out, pblk, blocks, df in rows:
        args = ", ".join(
            OPERAND[k].format(b) if "{}" in OPERAND[k] else OPERAND[k]
            for k, b in zip(kinds, blocks)
        )
        if df:
            dfs.append((coef, spec, kinds, blocks, out, pblk))
            continue
        call = f"einsum('{spec}', {args})"
        if pblk is None:
            lines.append(f'        {out_name}["{out}"] += {coef_src(coef)} * {call}')
        else:
            perm = "(1, 0)" if len(out) == 2 else "(1, 0, 3, 2)"
            lines.append(f"        _t = {coef_src(coef)} * {call}")
            lines.append(f'        {out_name}["{out}"] += _t')
            lines.append(f'        {out_name}["{pblk}"] += _t.transpose{perm}')
    return lines, dfs


def emit_kernel(name, h2_od):
    terms = TERMS[name]
    out_name = "C1" if name.endswith("C1") else "C2"
    sig_h2 = "F" if name.startswith("H1") else "V"
    amps = [
        k for k in ("T1", "T2", "S2") if any(k in kinds for _, _, kinds, _ in terms)
    ]
    fn = name + ("_od" if (h2_od and name in HAS_H2) else "")
    args = ", ".join([out_name, sig_h2] + amps + ["cumulants"])
    body, all_dfs = [], []
    for coef, spec, kinds, pair in terms:
        lines, dfs = emit_term(coef, spec, kinds, pair, h2_od, out_name)
        body.extend(lines)
        all_dfs.extend(dfs)
    if all_dfs:
        body.append("        # blocks with three or more virtual indices, rebuilt")
        body.append("        # from the three-index integrals rather than stored")
        for coef, spec, kinds, blocks, out, pblk in all_dfs:
            ops = ", ".join(
                OPERAND[k].format(b) if "{}" in OPERAND[k] else OPERAND[k]
                for k, b in zip(kinds, blocks)
                if k != "H2"
            )
            h2blk = blocks[kinds.index("H2")]
            body.append(
                f'        self._df({out_name}, "{out}", '
                f'{"None" if pblk is None else chr(34) + pblk + chr(34)}, '
                f'{coef_src(coef)}, "{spec}", {kinds.index("H2")}, '
                f'"{h2blk}", B, [{ops}])'
            )
    return fn, args, body, len(all_dfs)


def main():
    header = open("codegen/kernel_header.py").read()
    total = 0
    chunks = []
    for name in TERMS:
        for h2_od in (False, True):
            if h2_od and name not in HAS_H2:
                continue
            fn, args, body, ndf = emit_kernel(name, h2_od)
            nterms = sum(1 for l in body if "einsum(" in l or "self._df" in l)
            if ndf:
                # needs the scratch sizing and loop bounds that live on the helper
                head = f"    def {fn}(self, {args}, scale=1.0, B=None):"
            else:
                head = f"    @staticmethod\n    def {fn}({args}, scale=1.0):"
            pre = [
                f"    # {nterms} lines",
                "",
                "        g1 = cumulants['gamma1']",
                "        e1 = cumulants['eta1']",
                "        l2 = cumulants['lambda2']",
                "",
            ]
            chunks.append("\n".join([pre[0], head] + pre[2:] + body))
            total += sum(1 for l in body if "einsum(" in l or "self._df" in l)
    print(f"# generated: {total} elementary contractions", file=sys.stderr)
    def fmt(name, seq):
        """Emit a tuple the way black would, so the file needs no reformatting."""
        one_line = f"{name} = ({', '.join(chr(34) + x + chr(34) for x in seq)})"
        if len(one_line) <= 88:
            return one_line + "\n"
        body = "".join(f'    "{x}",\n' for x in seq)
        return f"{name} = (\n{body})\n"

    labels = (
        "    # fmt: on\n\n\n"
        + fmt("ALL1_LABELS", ALL1)
        + fmt("OD2_LABELS", OD2)
        + fmt("T2_LABELS", T2_SUPPORT)
        + fmt("T1_LABELS", T1_SUPPORT)
        + "_DF_TARGET = 250_000\n"
    )
    src = header + "\n" + "\n\n".join(chunks) + "\n" + labels
    return src.rstrip("\n") + "\n"


if __name__ == "__main__":
    # written rather than piped: `conda run` appends a newline to captured
    # stdout, which would leave the file needing a reformat every time
    with open(OUT, "w") as fh:
        fh.write(main())
