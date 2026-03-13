import numpy as np

from forte2 import System, State, MCOptimizer
from forte2.orbitals.semicanonicalizer import Semicanonicalizer
from forte2.scf import RHF, ROHF, UHF
from forte2.orbitals import write_molden
from forte2.system.basis_utils import ml_from_shell_index_cca
from forte2.helpers.comparisons import approx


def _molden_shell_permutation(l):
    m_to_internal_index = {
        ml_from_shell_index_cca(l, idx): idx for idx in range(2 * l + 1)
    }
    molden_m_sequence = [0]
    for m in range(1, l + 1):
        molden_m_sequence.extend((m, -m))
    return [m_to_internal_index[m] for m in molden_m_sequence]


def _molden_basis_permutation(basis):
    permutation = []
    offset = 0
    for ishell in range(basis.nshells):
        shell = basis[ishell]
        assert shell.is_pure
        permutation.extend(offset + idx for idx in _molden_shell_permutation(shell.l))
        offset += shell.size
    return np.asarray(permutation, dtype=int)


def _parse_mo_blocks(text):
    blocks = []
    current = None
    in_mo_section = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "[MO]":
            in_mo_section = True
            continue
        if not in_mo_section:
            continue

        if line.startswith("Sym="):
            if current is not None:
                blocks.append(current)
            current = {"sym": line.split("=", maxsplit=1)[1].strip(), "coeffs": []}
        elif line.startswith("Ene="):
            current["ene"] = float(line.split("=", maxsplit=1)[1])
        elif line.startswith("Spin="):
            current["spin"] = line.split("=", maxsplit=1)[1].strip()
        elif line.startswith("Occup="):
            current["occup"] = float(line.split("=", maxsplit=1)[1])
        else:
            idx_str, coeff_str = line.split()
            current["coeffs"].append((int(idx_str), float(coeff_str)))

    if current is not None:
        blocks.append(current)

    return blocks


def _expected_mcopt_energies(mc):
    orig_to_contig = np.asarray(mc.mo_space.orig_to_contig, dtype=int)

    if mc.final_orbital == "original":
        return np.diag(mc.orb_opt.Fock)[orig_to_contig]

    semi = Semicanonicalizer(
        mo_space=mc.mo_space,
        system=mc.system,
        mix_inactive=False,
        mix_active=False,
    )
    C_contig = mc.C[0][:, orig_to_contig].copy()
    semi.semi_canonicalize(g1=mc.make_average_1rdm(), C_contig=C_contig)
    return semi.eps_semican[orig_to_contig]


def test_molden_writer_rhf(tmp_path):
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)
    rhf.run()

    path = tmp_path / "water.molden"
    write_molden(rhf, path)

    assert path.is_file()

    text = path.read_text()
    assert "[Molden Format]" in text
    assert "[Atoms] AU" in text
    assert "[GTO]" in text
    assert "[MO]" in text
    assert "[5D]" in text

    blocks = _parse_mo_blocks(text)
    assert len(blocks) == system.nmo

    occ_block = blocks[0]
    vir_block = blocks[rhf.ndocc]
    assert occ_block["sym"] == rhf.irrep_labels[0]
    assert occ_block["ene"] == approx(rhf.eps[0][0])
    assert occ_block["spin"] == "Alpha"
    assert occ_block["occup"] == approx(2.0)
    assert vir_block["sym"] == rhf.irrep_labels[rhf.ndocc]
    assert vir_block["ene"] == approx(rhf.eps[0][rhf.ndocc])
    assert vir_block["spin"] == "Alpha"
    assert vir_block["occup"] == approx(0.0)

    d_shell_start = None
    offset = 0
    for ishell in range(system.basis.nshells):
        shell = system.basis[ishell]
        if shell.l == 2:
            d_shell_start = offset
            break
        offset += shell.size

    assert d_shell_start is not None

    d_slice = slice(d_shell_start, d_shell_start + 5)
    imo = int(np.argmax(np.linalg.norm(rhf.C[0][d_slice, :], axis=0)))

    expected_perm = _molden_basis_permutation(system.basis)
    expected_coeff = rhf.C[0][expected_perm, imo]
    parsed_indices = np.asarray([idx for idx, _ in blocks[imo]["coeffs"]], dtype=int)
    parsed_coeff = np.asarray([coeff for _, coeff in blocks[imo]["coeffs"]], dtype=float)

    assert np.array_equal(parsed_indices, np.arange(1, system.nbf + 1))
    assert parsed_coeff == approx(expected_coeff)


def test_molden_writer_rohf(tmp_path):
    xyz = """
    H 0 0 0
    H 0 0 1.4
    """

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    rohf = ROHF(charge=1, ms=0.5)(system)
    rohf.run()

    path = tmp_path / "h2plus_rohf.molden"
    write_molden(rohf, path)

    blocks = _parse_mo_blocks(path.read_text())
    assert len(blocks) == system.nmo
    assert blocks[0]["spin"] == "Alpha"
    assert blocks[0]["occup"] == approx(1.0)
    assert blocks[1]["occup"] == approx(0.0)
    assert blocks[0]["ene"] == approx(rohf.eps[0][0])
    assert blocks[1]["ene"] == approx(rohf.eps[0][1])


def test_molden_writer_uhf(tmp_path):
    xyz = """
    H 0 0 0
    H 0 0 1.4
    """

    system = System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit"
    )
    uhf = UHF(charge=1, ms=0.5)(system)
    uhf.run()

    path = tmp_path / "h2plus_uhf.molden"
    write_molden(uhf, path)

    blocks = _parse_mo_blocks(path.read_text())
    assert len(blocks) == 2 * system.nmo

    alpha_blocks = blocks[: system.nmo]
    beta_blocks = blocks[system.nmo :]

    assert all(block["spin"] == "Alpha" for block in alpha_blocks)
    assert all(block["spin"] == "Beta" for block in beta_blocks)
    assert alpha_blocks[0]["occup"] == approx(1.0)
    assert alpha_blocks[1]["occup"] == approx(0.0)
    assert beta_blocks[0]["occup"] == approx(0.0)
    assert beta_blocks[1]["occup"] == approx(0.0)
    assert alpha_blocks[0]["ene"] == approx(uhf.eps[0][0])
    assert beta_blocks[0]["ene"] == approx(uhf.eps[1][0])


def test_molden_writer_mcopt(tmp_path):
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1])(rhf)
    mc.run()

    path = tmp_path / "h2_mcopt.molden"
    write_molden(mc, path)

    blocks = _parse_mo_blocks(path.read_text())
    assert len(blocks) == system.nmo
    assert all(block["spin"] == "Alpha" for block in blocks)
    parsed_energies = np.asarray([block["ene"] for block in blocks], dtype=float)
    assert parsed_energies == approx(_expected_mcopt_energies(mc))

    expected_occupations = np.zeros(system.nmo, dtype=float)
    expected_occupations[np.asarray(mc.mo_space.docc_indices, dtype=int)] = 2.0
    expected_occupations[np.asarray(mc.mo_space.active_indices, dtype=int)] = np.diag(
        mc.make_average_1rdm()
    )
    parsed_occupations = np.asarray([block["occup"] for block in blocks], dtype=float)
    assert parsed_occupations == approx(expected_occupations)


def test_molden_writer_mcopt_original_orbitals(tmp_path):
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=2, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        final_orbital="original",
    )(rhf)
    mc.run()

    path = tmp_path / "h2_mcopt_original.molden"
    write_molden(mc, path)

    blocks = _parse_mo_blocks(path.read_text())
    assert len(blocks) == system.nmo
    parsed_energies = np.asarray([block["ene"] for block in blocks], dtype=float)
    assert parsed_energies == approx(_expected_mcopt_energies(mc))
