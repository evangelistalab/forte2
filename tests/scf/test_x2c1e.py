import numpy as np
import pytest

from forte2 import System, X2CParams
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.scf import RHF, GHF, UHF
from forte2.helpers.comparisons import approx
from forte2.system import BSE_AVAILABLE
from forte2.data import EH_TO_WN, EH_TO_EV


def _random_hcore_density(size, complex_values=False):
    rng = np.random.default_rng(8675309)
    density = rng.standard_normal((size, size))
    if complex_values:
        density = density + 1j * rng.standard_normal((size, size))
    density = 0.5 * (density + density.conj().T)
    return density / np.linalg.norm(density)


def _four_point_hcore_gradient_component(
    system_factory, coordinate, density, step=1.0e-4
):
    values = [
        np.einsum(
            "mn,nm->",
            system_factory(coordinate + scale * step).ints_hcore(),
            density,
        ).real
        for scale in (-2.0, -1.0, 1.0, 2.0)
    ]
    return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * step)


@pytest.mark.parametrize("x2c_type", ["sf", "so"])
@pytest.mark.parametrize(
    "use_gaussian_charges",
    [
        pytest.param(False, id="point"),
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not LIBCINT_AVAILABLE, reason="Libcint is not available"
            ),
            id="gaussian",
        ),
    ],
)
def test_x2c_hcore_gradient_finite_difference(x2c_type, use_gaussian_charges):
    def make_system(z):
        return System(
            xyz=f"O 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c=X2CParams(x2c_type=x2c_type, x2c_model="1e"),
            use_gaussian_charges=use_gaussian_charges,
            minao_basis_set=None,
        )

    system = make_system(1.5)
    size = system.nbf if x2c_type == "sf" else 2 * system.nbf
    density = _random_hcore_density(size, complex_values=x2c_type == "so")
    analytical = system.x2c_helper.hcore_gradient(density)[1, 2]
    numerical = _four_point_hcore_gradient_component(make_system, 1.5, density)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)


@pytest.mark.parametrize("snso_type", ["boettger", "dc", "dcb", "row-dependent"])
def test_snso_x2c_hcore_gradient_finite_difference(snso_type):
    def make_system(z):
        return System(
            xyz=f"S 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type=snso_type),
            minao_basis_set=None,
        )

    system = make_system(1.5)
    density = _random_hcore_density(2 * system.nbf, complex_values=True)
    analytical = system.x2c_helper.hcore_gradient(density)[1, 2]
    numerical = _four_point_hcore_gradient_component(make_system, 1.5, density)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)


def test_x2c_hcore_gradient_with_truncated_overlap_space():
    def make_system(z):
        return System(
            xyz=f"O 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
            overlap_ortho_rtol=1.0e-3,
            minao_basis_set=None,
        )

    system = make_system(1.5)
    assert system.x2c_helper.orth_info["n_discarded"] == 1
    density = _random_hcore_density(system.nbf)
    analytical = system.x2c_helper.hcore_gradient(density)[1, 2]
    numerical = _four_point_hcore_gradient_component(make_system, 1.5, density)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)


def test_x2c_helper_tracks_spinor_upcaster_override():
    def make_system(x2c):
        return System(
            xyz="S 0 0 0\nH 0 0 1.5",
            basis_set="sto-3g",
            unit="bohr",
            x2c=x2c,
            minao_basis_set=None,
        )

    overridden = make_system(X2CParams(x2c_type="sf", x2c_model="1e"))
    overridden.x2c = X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent")
    overridden._init_x2c()
    reference = make_system(
        X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent")
    )

    assert overridden.x2c_helper.hcore_x2c() == pytest.approx(
        reference.x2c_helper.hcore_x2c(), abs=1.0e-12
    )
    density = _random_hcore_density(2 * overridden.nbf, complex_values=True)
    assert overridden.x2c_helper.hcore_gradient(density) == pytest.approx(
        reference.x2c_helper.hcore_gradient(density), abs=1.0e-11
    )


def test_sfx2c1e():
    escf = -5192.021043979554
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_sfx2c1e_with_gaussian_charges():
    escf = -5192.003129895058
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
        use_gaussian_charges=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


@pytest.mark.skip(reason="This test cannot be reliably reproduced.")
def test_lindep_sfx2c1e():
    # psi4's x2c actually doesn't handle this case correctly
    # pyscf gives -4.071624245913899, so we need to investigate further
    # mol = pyscf.gto.M(
    #     atom=["H 0 0 %f" % i for i in range(10)],
    #     unit="Bohr",
    #     basis_set="aug-cc-pvdz",
    #     symmetry=False,
    # )
    # mf = pyscf.scf.RHF(mol).density_fit("cc-pvqz-jkfit").x2c()
    # mf = pyscf.scf.addons.remove_linear_dep_(mf, threshold=2e-7, lindep=1e-10)
    # mf.kernel()
    erhf = -4.071623764438

    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = System(
        xyz=xyz,
        basis_set="aug-cc-pvdz",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
        overlap_ortho_rtol=2e-7,
    )

    scf = RHF(charge=0, e_tol=1e-10, d_tol=1e-8)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert scf.nbf == 90
    assert scf.nmo == 81


def test_sox2c1e_water():
    eghf = -76.08194686989626
    xyz = """
    O 0 0 0
    H 0 -0.757 0.587
    H 0 0.757 0.587
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(eghf)


def test_snso_shell_to_atom_mapping():
    import numpy as np
    from forte2.x2c.x2c import X2CHelper

    system = System(
        xyz="Ne 0 0 0; Ar 0 0 3.0",
        basis_set="cc-pVTZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="dcb"),
    )
    helper = X2CHelper(system)
    nbf = len(helper.xbasis)
    scaled = helper._apply_snso_scaling(np.ones((nbf, nbf)))

    # Independent reference using the correct shell-index -> atom mapping.
    b = helper.xbasis
    shell_first = np.array([x[0] for x in b.center_first_and_last_shell])
    atoms = [a[0] for a in system.atoms]
    center = lambda ish: int(np.searchsorted(shell_first, ish, side="right") - 1)
    Ql = np.array([0.0, 2.97, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0])
    ref = np.ones((nbf, nbf))
    iptr = 0
    for ish in range(b.nshells):
        isz = b[ish].size
        li = int(b[ish].l)
        if li == 0:
            iptr += isz
            continue
        Zi = atoms[center(ish)]
        jptr = 0
        for jsh in range(b.nshells):
            jsz = b[jsh].size
            lj = int(b[jsh].l)
            if lj == 0:
                jptr += jsz
                continue
            Zj = atoms[center(jsh)]
            factor = 1 - np.sqrt(Ql[li] * Ql[lj] / (Zi * Zj))
            ref[iptr : iptr + isz, jptr : jptr + jsz] *= factor
            jptr += jsz
        iptr += isz

    assert np.allclose(scaled, ref)


def test_boettger_hbr():
    xyz = """
    H 0 0 0
    Br 0 0 1.4
    """

    system = System(
        xyz=xyz,
        basis_set={"Br": "decon-aug-cc-pvdz", "default": "cc-pvtz"},
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="dcb"),
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert EH_TO_WN * (
        scf.eps[0][scf.nel - 2] - scf.eps[0][scf.nel - 3]
    ) == pytest.approx(2953.193840779559, abs=1e-4)


def test_so_from_sf_water():
    euhf = -75.711680104122
    eghf = -75.7116858952105
    xyz = """
    O 0 0 0
    H 0 -0.757 0.587
    H 0 0.757 0.587
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=X2CParams(x2c_type="sf", x2c_model="1e"),
    )
    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf_so = GHF(charge=1)(system)
    mos_2c = scf.mos.to_spinorbital_basis()
    scf_so.C = mos_2c.C
    scf_so.run()
    assert scf_so.E == approx(eghf)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_sox2c1e_sc():
    l23_ref = 4.39507729290027
    xyz = "Sc 0 0 0"
    system = System(
        xyz=xyz,
        basis_set="sapporo-dkh3-dzp-2012-diffuse",
        auxiliary_basis_set="def2-universal-jkfit",
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="row-dependent"),
    )
    scf = GHF(charge=3)(system)
    scf.run()
    l23_splitting = EH_TO_EV * (scf.eps[0][6] - scf.eps[0][5])
    assert l23_splitting == pytest.approx(l23_ref, abs=1e-5)
