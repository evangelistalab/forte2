"""Tests for the Molden writer.

The central check is value-based: a small Molden reader written in this module
(independently of the writer, from the Molden conventions) evaluates the
written orbitals at points in space, and the result is compared with Forte2's
own ``ints.orbitals_on_grid``. Coefficient-level comparisons against the
writer's own permutation cannot detect convention errors; this can.
"""

import math
import re

import numpy as np
import pytest

from forte2 import AVAS, CI, CISolver, GHF, MCOptimizer, ROHF, RHF, State, System, UHF
from forte2.helpers.comparisons import approx
from forte2.lib import ints
from forte2.orbitals import molden_string, write_molden
from forte2.orbitals.semicanonicalizer import Semicanonicalizer

WATER = """
O 0.000000000000  0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369  0.489330954643
H 0.000000000000  0.711620616369  0.489330954643
"""
# No symmetry: x, y and z components are all distinguishable.
HOCL_TILTED = """
O   0.10 -0.20  0.05
H   0.95  0.15 -0.30
Cl -1.05  0.65  0.40
"""
CYCLOPROPENE = """
H   0.912650   0.000000   1.457504
H  -0.912650   0.000000   1.457504
H   0.000000  -1.585659  -1.038624
H   0.000000   1.585659  -1.038624
C   0.000000   0.000000   0.859492
C   0.000000  -0.651229  -0.499559
C   0.000000   0.651229  -0.499559
"""


# --------------------------------------------------------------------------- #
# Independent Molden reader (pure, segmented shells, up to g)
# --------------------------------------------------------------------------- #


def _solid_harmonics(l, x, y, z):
    """Real solid harmonics in Molden order, up to a constant per function."""
    r2 = x * x + y * y + z * z
    if l == 0:
        return [np.ones_like(x)]
    if l == 1:
        return [x, y, z]
    if l == 2:
        return [2 * z * z - x * x - y * y, x * z, y * z, x * x - y * y, x * y]
    if l == 3:
        return [
            z * (2 * z * z - 3 * x * x - 3 * y * y),
            x * (4 * z * z - x * x - y * y),
            y * (4 * z * z - x * x - y * y),
            z * (x * x - y * y),
            x * y * z,
            x * (x * x - 3 * y * y),
            y * (3 * x * x - y * y),
        ]
    if l == 4:
        return [
            35 * z**4 - 30 * z * z * r2 + 3 * r2 * r2,
            x * z * (7 * z * z - 3 * r2),
            y * z * (7 * z * z - 3 * r2),
            (x * x - y * y) * (7 * z * z - r2),
            x * y * (7 * z * z - r2),
            x * z * (x * x - 3 * y * y),
            y * z * (3 * x * x - y * y),
            x**4 - 6 * x * x * y * y + y**4,
            x * y * (x * x - y * y),
        ]
    raise ValueError(l)


def _angular_scale(l):
    """Factors giving each harmonic the angular norm of the axial x^l."""
    ct, wt = np.polynomial.legendre.leggauss(16)
    phi = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
    st = np.sqrt(1 - ct**2)
    x = np.outer(st, np.cos(phi)).ravel()
    y = np.outer(st, np.sin(phi)).ravel()
    z = np.repeat(ct, phi.size)
    w = np.repeat(wt, phi.size)
    axial = np.sum(w * x ** (2 * l))
    return np.array(
        [math.sqrt(axial / np.sum(w * f * f)) for f in _solid_harmonics(l, x, y, z)]
    )


def _parse_molden(text):
    sections = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*(.*)$", line)
        if m:
            name = m.group(1).strip().upper()
            sections[name] = {"option": m.group(2).strip(), "lines": []}
        elif name is not None:
            sections[name]["lines"].append(line)
    assert sections["ATOMS"]["option"].upper() == "AU"
    atoms = []
    for line in sections["ATOMS"]["lines"]:
        if line.strip():
            p = line.split()
            atoms.append(np.array([float(v) for v in p[3:6]]))
    shells = []
    lines = sections["GTO"]["lines"]
    i = 0
    atom = None
    labels = "spdfg"
    while i < len(lines):
        p = lines[i].split()
        i += 1
        if not p:
            continue
        if len(p) == 2 and p[0].isdigit():
            atom = int(p[0]) - 1
            continue
        l = labels.index(p[0].lower())
        nprim = int(p[1])
        prims = [
            tuple(float(v.replace("D", "E")) for v in lines[i + k].split())
            for k in range(nprim)
        ]
        i += nprim
        shells.append(
            (atom, l, np.array([q[0] for q in prims]), np.array([q[1] for q in prims]))
        )
    mos = []
    cur = None
    for line in sections["MO"]["lines"]:
        s = line.strip()
        if not s:
            continue
        if "=" in s:
            key, val = (t.strip() for t in s.split("=", 1))
            if key.upper() == "SYM":
                cur = {"sym": val, "C": {}}
                mos.append(cur)
            elif key.upper() == "ENE":
                cur["ene"] = float(val)
            elif key.upper() == "SPIN":
                cur["spin"] = val
            elif key.upper() == "OCCUP":
                cur["occup"] = float(val)
        else:
            k, v = s.split()
            cur["C"][int(k)] = float(v)
    nao = sum(2 * l + 1 for _, l, _, _ in shells)
    C = np.zeros((nao, len(mos)))
    for j, mo in enumerate(mos):
        for k, v in mo["C"].items():
            C[k - 1, j] = v
    return {"sections": sections, "atoms": atoms, "shells": shells, "mos": mos, "C": C}


def _eval_molden_aos(parsed, points):
    """AO values in Molden order. Primitives are normalized; contractions are not
    renormalized, so normalization errors in the file show up as errors here."""
    cols = []
    for atom, l, alpha, coef in parsed["shells"]:
        d = points - parsed["atoms"][atom]
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        r2 = np.sum(d * d, axis=1)
        norm = np.sqrt(
            2.0**l
            * (2 * alpha) ** (l + 1.5)
            / (math.pi**1.5 * math.prod(range(2 * l - 1, 0, -2)))
        )
        radial = np.exp(-np.outer(r2, alpha)) @ (coef * norm)
        for f, k in zip(_solid_harmonics(l, x, y, z), _angular_scale(l)):
            cols.append(k * f * radial)
    return np.array(cols)


def _points(system, n=600, pad=2.5, seed=7):
    xyz = np.array([c for _, c in system.atoms])
    rng = np.random.default_rng(seed)
    lo, hi = xyz.min(axis=0) - pad, xyz.max(axis=0) + pad
    return lo + (hi - lo) * rng.random((n, 3))


def _forte2_on_points(system, C, points):
    """Evaluate orbitals with ints.orbitals_on_grid, one 1x1x1 'grid' per point."""
    out = np.empty((points.shape[0], C.shape[1]))
    axes = np.eye(3).tolist()
    for i, p in enumerate(points):
        out[i] = np.asarray(
            ints.orbitals_on_grid(system.basis, C, list(p), [1, 1, 1], axes)
        )[0]
    return out.T


def _assert_molden_matches(system, C, text, tol=1e-9):
    parsed = _parse_molden(text)
    assert parsed["C"].shape[0] == system.nbf
    pts = _points(system)
    ref = _forte2_on_points(system, np.ascontiguousarray(C), pts)
    test = parsed["C"][:, : C.shape[1]].T @ _eval_molden_aos(parsed, pts)
    scale = np.abs(ref).max(axis=1, keepdims=True)
    err = np.abs(test - ref) / scale
    assert (
        err.max() < tol
    ), f"max relative error {err.max():.2e} (worst MO {np.argmax(err.max(axis=1))})"
    return parsed


# --------------------------------------------------------------------------- #
# Value-based tests
# --------------------------------------------------------------------------- #


def test_molden_identity_basis_through_g_shells():
    """Every AO, s through g, on two atoms that are not at the origin."""
    system = System(
        xyz="Ne 0.31 -0.22 0.13\nHe 1.45 0.71 -0.93",
        basis_set="cc-pvqz",
        auxiliary_basis_set="def2-universal-jkfit",
    )
    assert max(system.basis[i].l for i in range(system.basis.nshells)) == 4
    C = np.eye(system.nbf)
    text = molden_string(system, C=C)
    assert "[5D]" in text and "[7F]" in text and "[9G]" in text
    _assert_molden_matches(system, C, text)


@pytest.mark.parametrize(
    "xyz, basis",
    [(WATER, "cc-pvdz"), (HOCL_TILTED, "cc-pvtz")],
    ids=["water-ccpvdz", "hocl-tilted-ccpvtz"],
)
def test_molden_rhf_orbitals_match_forte2(xyz, basis, tmp_path):
    system = System(xyz=xyz, basis_set=basis, auxiliary_basis_set="cc-pvtz-jkfit")
    rhf = RHF(charge=0)(system)
    rhf.run()
    path = write_molden(rhf, tmp_path / "rhf.molden")
    parsed = _assert_molden_matches(system, rhf.mos.C[0], path.read_text())

    mos = parsed["mos"]
    assert len(mos) == system.nmo
    assert [m["occup"] for m in mos] == approx(
        [2.0] * rhf.ndocc + [0.0] * (system.nmo - rhf.ndocc)
    )
    assert [m["ene"] for m in mos] == approx(rhf.eps[0])
    assert [m["sym"] for m in mos] == list(rhf.mos.irrep_labels[0])
    assert all(m["spin"] == "Alpha" for m in mos)


def test_molden_uhf_both_spin_blocks(tmp_path):
    system = System(xyz=WATER, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    uhf = UHF(charge=1, ms=0.5)(system)
    uhf.run()
    text = molden_string(uhf)
    parsed = _parse_molden(text)
    nmo = system.nmo
    assert len(parsed["mos"]) == 2 * nmo
    alpha, beta = parsed["mos"][:nmo], parsed["mos"][nmo:]
    assert all(m["spin"] == "Alpha" for m in alpha) and all(
        m["spin"] == "Beta" for m in beta
    )
    assert sum(m["occup"] for m in alpha) == approx(uhf.na)
    assert sum(m["occup"] for m in beta) == approx(uhf.nb)
    assert [m["ene"] for m in beta] == approx(uhf.eps[1])

    pts = _points(system)
    aos = _eval_molden_aos(parsed, pts)
    for block, C in (
        (slice(0, nmo), uhf.mos.C[0]),
        (slice(nmo, 2 * nmo), uhf.mos.C[1]),
    ):
        ref = _forte2_on_points(system, np.ascontiguousarray(C), pts)
        test = parsed["C"][:, block].T @ aos
        assert np.abs(test - ref).max() / np.abs(ref).max() < 1e-9


def test_molden_rohf_occupations(tmp_path):
    system = System(xyz=WATER, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rohf = ROHF(charge=1, ms=0.5)(system)
    rohf.run()
    parsed = _assert_molden_matches(system, rohf.mos.C[0], molden_string(rohf))
    occ = [m["occup"] for m in parsed["mos"]]
    assert occ[: rohf.nb] == approx([2.0] * rohf.nb)
    assert occ[rohf.nb : rohf.na] == approx([1.0] * (rohf.na - rohf.nb))
    assert sum(occ) == approx(rohf.na + rohf.nb)


def _cyclopropene_chain(final_orbitals="semicanonical"):
    system = System(
        xyz=CYCLOPROPENE, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    rhf = RHF(charge=0, e_tol=1e-10)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    mc = MCOptimizer(
        CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0)),
        final_orbitals=final_orbitals,
    )(avas)
    mc.run()
    return rhf, avas, mc


def test_molden_avas_orbitals():
    rhf, avas, mc = _cyclopropene_chain()
    parsed = _assert_molden_matches(avas.system, avas.mos.C[0], molden_string(avas))
    occ = np.array([m["occup"] for m in parsed["mos"]])
    # AVAS rotates occupied and virtual orbitals separately: nominal
    # occupations stay 2 or 0 and the electron count is conserved.
    assert np.all(np.isclose(occ, 0.0, atol=1e-8) | np.isclose(occ, 2.0, atol=1e-8))
    assert occ.sum() == approx(rhf.nel)
    assert occ[avas.mo_space.docc_indices] == approx(2.0)
    # Energies are the reference Fock diagonal in the AVAS orbitals.
    C, F = avas.mos.C[0], rhf.F[0]
    assert [m["ene"] for m in parsed["mos"]] == approx(np.diag(C.T @ F @ C))


@pytest.mark.parametrize("final_orbitals", ["semicanonical", "natural", "original"])
def test_molden_mcoptimizer(final_orbitals):
    rhf, avas, mc = _cyclopropene_chain(final_orbitals)
    parsed = _assert_molden_matches(mc.system, mc.mos.C[0], molden_string(mc))
    mos = parsed["mos"]
    g1 = mc.make_average_rdm(1)
    occ = np.array([m["occup"] for m in mos])
    assert occ[mc.mo_space.docc_indices] == approx(2.0)
    assert occ[mc.mo_space.active_indices] == approx(np.diag(g1))
    assert occ.sum() == approx(rhf.nel)
    if final_orbitals == "natural":
        assert occ[mc.mo_space.active_indices] == approx(np.linalg.eigvalsh(g1)[::-1])

    if final_orbitals == "semicanonical":
        # Independent path: the semicanonicalizer's own eigenvalues.
        o2c = np.asarray(mc.mo_space.orig_to_contig)
        semi = Semicanonicalizer(
            mo_space=mc.mo_space,
            system=mc.system,
            irrep_indices=np.asarray(mc.mos.irrep_indices[0])[o2c],
            mix_inactive=False,
            mix_active=False,
            do_active=True,
        )
        semi.semi_canonicalize(g1=g1, C_contig=mc.mos.C[0][:, o2c].copy())
        expected = semi.eps_semican[np.asarray(mc.mo_space.contig_to_orig)]
        assert [m["ene"] for m in mos] == approx(expected)


def test_molden_ci_driver():
    system = System(
        xyz=CYCLOPROPENE, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    rhf = RHF(charge=0, e_tol=1e-10)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    ci = CI(CISolver(State(nel=rhf.nel, multiplicity=1, ms=0.0)))(avas)
    ci.run()
    parsed = _assert_molden_matches(system, ci.mos.C[0], molden_string(ci))
    occ = np.array([m["occup"] for m in parsed["mos"]])
    assert occ[ci.mo_space.active_indices] == approx(np.diag(ci.make_average_rdm(1)))


def test_molden_system_with_unrestricted_arrays():
    system = System(xyz=WATER, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rng = np.random.default_rng(3)
    Ca, Cb = rng.normal(size=(2, system.nbf, 4))
    text = molden_string(
        system,
        C=[Ca, Cb],
        energies=[np.arange(4.0), -np.arange(4.0)],
        occupations=[np.ones(4), np.zeros(4)],
        symmetries=[["a1"] * 4, ["b2"] * 4],
        title="random orbitals",
    )
    assert "[Title]\nrandom orbitals\n" in text
    parsed = _parse_molden(text)
    assert [m["spin"] for m in parsed["mos"]] == ["Alpha"] * 4 + ["Beta"] * 4
    assert [m["sym"] for m in parsed["mos"]] == ["a1"] * 4 + ["b2"] * 4
    pts = _points(system)
    ref = _forte2_on_points(system, np.ascontiguousarray(np.hstack([Ca, Cb])), pts)
    test = parsed["C"].T @ _eval_molden_aos(parsed, pts)
    assert np.abs(test - ref).max() / np.abs(ref).max() < 1e-9


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_molden_rejects_non_method(tmp_path):
    with pytest.raises(TypeError, match="Method or System"):
        write_molden(object(), tmp_path / "invalid.molden")


def test_molden_requires_system_and_mos(tmp_path):
    method = RHF(charge=0)
    method.provides.remove("mos")
    with pytest.raises(RuntimeError, match="does not provide required data: mos"):
        write_molden(method, tmp_path / "missing_mos.molden")


def test_molden_requires_executed_method():
    system = System(
        xyz=WATER, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    with pytest.raises(RuntimeError, match="has not been run"):
        molden_string(RHF(charge=0)(system))


def test_molden_argument_combinations():
    system = System(
        xyz=WATER, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    with pytest.raises(TypeError, match="C is required"):
        molden_string(system)
    rhf = RHF(charge=0)(system)
    rhf.run()
    with pytest.raises(TypeError, match="only be combined with a System"):
        molden_string(rhf, C=rhf.mos.C[0])


def test_molden_rhf_on_system_shared_with_ghf():
    """Binding GHF flips the shared System's two_component flag in place; the
    already-run RHF still writes, because the writer checks the method's flag."""
    system = System(
        xyz="H 0 0 0\nH 0 0 0.74",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
    )
    rhf = RHF(charge=0)(system)
    rhf.run()
    GHF(charge=0)(system)
    assert system.two_component
    _assert_molden_matches(system, rhf.mos.C[0], molden_string(rhf))


def test_molden_rejects_spinorbitals():
    system = System(
        xyz="H 0 0 0\nH 0 0 0.74",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
    )
    ghf = GHF(charge=0)(system)
    ghf.run()
    with pytest.raises(NotImplementedError, match="2ccube"):
        molden_string(ghf)
