import pytest
import numpy as np

from forte2 import System
from forte2.data import ATOM_SYMBOL_TO_Z
from forte2.system.build_basis import BSE_AVAILABLE, _load_basis

if BSE_AVAILABLE:
    from basis_set_exchange import manip


def _exponents_by_l(shells):
    res = {}
    for l, exponents in shells:
        res.setdefault(l, []).extend(exponents)
    return {l: sorted(v) for l, v in res.items()}


def _forte2_autoaux(system):
    positions = np.array([coords for _, coords in system.atoms])
    shells = [[] for _ in system.atoms]
    for i in range(system.auxiliary_basis.nshells):
        sh = system.auxiliary_basis[i]
        iatom = int(np.argmin(np.linalg.norm(positions - sh.center, axis=1)))
        shells[iatom].append((sh.l, sh.exponents))
    return [_exponents_by_l(s) for s in shells]


def _bse_autoaux(basis_name, Z):
    # the Basis Set Exchange algorithm, applied to the orbital basis data forte2 uses
    orbital = {
        "name": basis_name,
        "version": "0",
        "revision_description": "",
        "elements": {str(Z): {"electron_shells": _load_basis(basis_name, Z)}},
    }
    aux = manip.autoaux_basis(orbital)["elements"][str(Z)]["electron_shells"]
    return _exponents_by_l(
        (sh["angular_momentum"][0], map(float, sh["exponents"])) for sh in aux
    )


@pytest.mark.skipif(not BSE_AVAILABLE, reason="basis_set_exchange not installed")
def test_autoaux_matches_bse():
    # cc-pvtz is packaged with forte2 and differs from the Basis Set Exchange
    # version for Li, Be, and Ca; 6-31g* has sp shells; ano-rcc-vtzp is generally
    # contracted; cc-pwcvtz has tight core-valence functions
    cases = [
        ("cc-pvtz", ["H", "Li", "Be", "O", "Ca", "Cr", "Kr"]),
        ("def2-qzvp", ["B", "Si", "Fe"]),
        ("6-31g*", ["H", "N", "K", "Cu"]),
        ("cc-pwcvtz", ["O", "S", "Zn"]),
        ("aug-cc-pvqz", ["H", "Cl"]),
        ("ano-rcc-vtzp", ["O", "Fe"]),
    ]
    for basis_name, elements in cases:
        xyz = "\n".join(f"{el} 0 0 {4.0 * i}" for i, el in enumerate(elements))
        system = System(
            xyz=xyz,
            basis_set=basis_name,
            auxiliary_basis_set=f"{basis_name}-autoaux",
            minao_basis_set=None,
        )
        for el, aux in zip(elements, _forte2_autoaux(system)):
            ref = _bse_autoaux(basis_name, ATOM_SYMBOL_TO_Z[el.upper()])
            assert aux.keys() == ref.keys(), (basis_name, el)
            for l in ref:
                assert len(aux[l]) == len(ref[l]), (basis_name, el, l)
                assert np.allclose(aux[l], ref[l], rtol=1e-12, atol=0)

    # In def2-tzvp, both d-shell bounds of H come from its single p primitive and
    # tie exactly, so the even-tempered series stops at its first exponent. Basis
    # Set Exchange resolves this tie by rounding error that differs between
    # platforms, so the expected shell is checked directly.
    system = System(
        xyz="H 0 0 0",
        basis_set="def2-tzvp",
        auxiliary_basis_set="def2-tzvp-autoaux",
        minao_basis_set=None,
    )
    (aux,) = _forte2_autoaux(system)
    assert aux[2] == [1.6]
