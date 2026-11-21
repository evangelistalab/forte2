from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

import forte2
from forte2.base_classes import ActiveSpaceDriver, Method, MO, X2CParams
from forte2.dsrg.utils import (
    cas_energy_given_RDMs,
    compute_t1_block,
    compute_t2_block,
    renormalize_V_block,
)
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.orbitals.aset import ASET
from forte2.orbitals.semicanonicalizer import Semicanonicalizer
from forte2.state import EmbeddingMOSpace, MOSpace


class DressedFockBuilder:
    """
    Fock and MO-integral builder backed by a dense four-index two-electron tensor
    with 4-fold Hermitian + particle-exchange symmetry:

        V_phys[p, q, r, s] = <pq | rs> = <qp | sr> = <rs | pq>^*

    Unlike a Cholesky/DF factorized Coulomb tensor, a downfolded effective
    two-electron interaction does not in general possess 8-fold Chemist's
    symmetry (<pq|rs> != <rq|ps>). This builder evaluates J, K, generalized Fock
    matrices, and transformed 4-index integrals directly from V_phys with exact
    4-fold symmetry.
    """

    has_dense_tei: bool = True

    def __init__(self, system, V_phys: NDArray):
        self.system = system
        self.V_phys = np.ascontiguousarray(V_phys)
        self.nbf = system.nbf

    def build_JK(self, C):
        J = []
        K = []
        for Ci in C:
            if Ci.shape[1] == 0:
                z = np.zeros((self.nbf, self.nbf), dtype=self.V_phys.dtype)
                J.append(z)
                K.append(z.copy())
                continue
            D = np.einsum("mi,ni->mn", Ci, Ci.conj(), optimize=True)
            Ji = np.einsum("prqs,sr->pq", self.V_phys, D, optimize=True)
            Ki = np.einsum("prsq,sr->pq", self.V_phys, D, optimize=True)
            J.append(Ji)
            K.append(Ki)
        return J, K

    def build_JK_generalized(self, C, g1):
        assert C.shape[1] == g1.shape[0], "C and g1 must have compatible dimensions"
        if C.shape[1] == 0:
            z = np.zeros((self.nbf, self.nbf), dtype=self.V_phys.dtype)
            return z, z.copy()
        D = np.einsum("ru,uv,sv->rs", C.conj(), g1, C, optimize=True)
        J = np.einsum("prqs,rs->pq", self.V_phys, D, optimize=True)
        K = np.einsum("prsq,rs->pq", self.V_phys, D, optimize=True)
        return J, K

    def build_core_fock(self, C_core, hcore=None):
        if hcore is None:
            hcore = self.system.ints_hcore()
        Jcore, Kcore = self.build_JK([C_core])
        return hcore + 2.0 * Jcore[0] - Kcore[0]

    def build_active_fock(self, C_act, g1):
        Jact, Kact = self.build_JK_generalized(C_act, g1)
        return Jact - 0.5 * Kact

    def build_generalized_fock(self, C_core, C_act, g1, hcore=None):
        return self.build_core_fock(C_core, hcore=hcore) + self.build_active_fock(
            C_act, g1
        )

    def two_electron_integrals_block(self, C):
        return np.einsum(
            "pqrs,pi,qj,rk,sl->ijkl",
            self.V_phys,
            C.conj(),
            C.conj(),
            C,
            C,
            optimize=True,
        )

    def two_electron_integrals_gen_block(self, C1, C2, C3, C4):
        return np.einsum(
            "pqrs,pi,qj,rk,sl->ijkl",
            self.V_phys,
            C1.conj(),
            C2.conj(),
            C3,
            C4,
            optimize=True,
        )


class DressedFragmentSystem:
    """
    System-like container holding an ASET(2) dressed fragment Hamiltonian

        Gbar = E^{A+B,[2]} + sum_{uv in A} hbar_{uv} E^u_v
                           + 1/2 sum_{uvxy in A} hbar_{uvxy} e^{uv}_{xy}

    in the orthonormal fragment MO basis of size N_A = dim(C_A) + dim(A_A) + dim(V_A).
    Compatible with downstream forte2 solvers that respect 4-fold Hermitian
    two-electron symmetry (``CISolver`` with default ``ci_algorithm="hz"``,
    ``SelectedCISolver`` with ``ci_algorithm="exact"``, ``MCOptimizer``, and
    ``DSRG_MRPT2``).
    """

    def __init__(
        self,
        scalar_energy: float,
        hcore: NDArray,
        V_phys: NDArray,
        nel_frag: int,
    ):
        self.nuclear_repulsion = float(scalar_energy)
        self.hcore = np.ascontiguousarray(hcore)
        self.V_phys = np.ascontiguousarray(V_phys)
        self.nbf = self.hcore.shape[0]
        self.nmo = self.nbf
        self.naux = 0
        self.Zsum = int(nel_frag)
        self.overlap = np.eye(self.nbf, dtype=self.hcore.dtype)
        self.Xorth = np.eye(self.nbf, dtype=self.hcore.dtype)
        self.two_component = False
        self.symmetry = False
        self.point_group = "C1"
        self.x2c = X2CParams()
        self.df_ortho_rtol = None
        self.cholesky_tei = False
        self.fock_builder = DressedFockBuilder(self, self.V_phys)
        self.fock_builder_corr = self.fock_builder

    def ints_overlap(self):
        return self.overlap

    def ints_hcore(self):
        return self.hcore

    def nuclear_repulsion_energy(self):
        return self.nuclear_repulsion

    def get_Xorth(self):
        return self.Xorth


def _regularized_denom(x: NDArray, s: float) -> NDArray:
    """Evaluate (1 - exp(-s * x^2)) / x stably near x = 0."""
    z = np.sqrt(s) * x
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(z) <= 1e-4
    if np.any(small):
        zs = z[small]
        z2 = zs * zs
        # (1 - exp(-z^2)) / z = z * (1 - z^2/2 + z^4/6 - z^6/24 + z^8/120)
        out[small] = (
            np.sqrt(s)
            * zs
            * (1.0 - z2 / 2.0 + z2 * z2 / 6.0 - (z2**3) / 24.0 + (z2**4) / 120.0)
        )
    if np.any(~small):
        xl = x[~small]
        out[~small] = (1.0 - np.exp(-s * xl * xl)) / xl
    return out


@dataclass
class ASET2(Method):
    """
    Second-Order Active-Space Embedding Theory (ASET(2)).

    Performs a one-shot second-order DSRG canonical transformation that decouples
    the fragment A = {C_A, A_A, V_A} from the environment B = {O_B, V_B},
    producing a dressed fragment Hamiltonian Gbar over the N_A fragment orbitals.

    The resulting ``DressedFragmentSystem``, identity ``mos``, and fragment ``mo_space``
    can be passed directly to any downstream solver (``CI``, ``SelectedCISolver``,
    ``MCOptimizer``, ``DSRG_MRPT2``).

    Parameters
    ----------
    flow_param : float, optional, default=0.5
        DSRG flow parameter s (in Eh^-2) for the ASET(2) downfolding.
    include_env_corr : bool, optional, default=True
        Whether to include pure-environment excitations (T1^B: O_B -> V_B and
        T2^B: O_B O_B -> V_B V_B, corresponding to delta E^B_{c,ASET(2)}).
        Set to False to zero all pure-environment amplitudes.
    skip_3_cumulant : bool, optional, default=True
        Whether to neglect the 3-body density cumulant contribution (Eq. 35 of
        Ref. [1]). Because lambda3 is nonzero only on A_A^6, setting this to
        False is inexpensive for typical active spaces.
    root : int | None, optional, default=None
        If None, use the state-averaged RDMs of the parent reference. If an
        integer is given, use the state-specific RDMs of that CI root and
        re-semicanonicalize the 5 embedding subspaces with its Fock matrix
        (Sec. 2.5 of Ref. [1]).

    Attributes
    ----------
    E_ref : float
        Reference energy <Psi_0 | bare H^{A+B} | Psi_0>.
    delta_E_AB : float
        Fragment-environment second-order correlation energy delta E^{AB}_{c,ASET(2)}.
    delta_E_B : float
        Pure-environment second-order correlation energy delta E^{B}_{c,ASET(2)}.
    E_aset2_ref : float
        Unrelaxed ASET(2) reference energy E_ref + delta_E_AB + delta_E_B.
    E_scalar : float
        Dressed scalar energy E^{A+B,[2]} entering Gbar.
    hbar1 : NDArray
        Dressed one-electron integrals hbar_u^{v,[2]} over fragment A (shape (N_A, N_A)).
    hbar2 : NDArray
        Dressed two-electron integrals hbar_{uv}^{xy,[2]} over fragment A (shape (N_A,)*4).
    nel_frag : int
        Number of electrons in fragment A (excluding frozen core and O_B).

    References
    ----------
    .. [1] N. He, C. Li, and F. A. Evangelista, "Second-Order Active-Space Embedding Theory",
           J. Chem. Theory Comput. 2022, 18, 1527-1541.
    """

    flow_param: float = 0.5
    include_env_corr: bool = True
    skip_3_cumulant: bool = True
    root: int | None = None

    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        assert self.flow_param >= 0.0, f"flow_param must be >= 0, got {self.flow_param}"
        self.requires = {"system", "mos", "mo_space"}
        self.provides = {"system", "mos", "mo_space"}
        self.requires_attrs.update({"two_component": False})

    def __call__(self, parent_method):
        if isinstance(parent_method, ASET):
            pass
        elif isinstance(parent_method, ActiveSpaceDriver) and isinstance(
            parent_method.parent_method, ASET
        ):
            pass
        else:
            raise TypeError(
                "ASET2 parent must be ASET or a CI/MCOptimizer driver whose parent is ASET, "
                f"got {type(parent_method).__name__}."
            )
        self._register_parent_method(parent_method)
        return self

    def _resolve_upstream(self):
        """Return (aset_obj, ci_driver_or_None, full_system, full_mos, emb_space)."""
        if isinstance(self.parent_method, ASET):
            aset = self.parent_method
            ci_driver = (
                aset.parent_method
                if isinstance(aset.parent_method, ActiveSpaceDriver)
                else None
            )
            full_mos = aset.mos.copy()
            emb_space = aset.emb_space
        else:
            ci_driver = self.parent_method
            aset = ci_driver.parent_method
            full_mos = ci_driver.mos.copy()
            # Rebuild EmbeddingMOSpace in case ci_driver defined an active space
            # inside fragment A (e.g., RHF -> ASET -> CI(2e, 2o) -> ASET2).
            aset_emb = aset.emb_space
            frag_indices = set(
                aset.partition["index_A_occ"]
                + [i for g in aset.partition["active_orbitals"] for i in g]
                + aset.partition["index_A_vir"]
            )
            ci_ms = ci_driver.mo_space
            A_core = [i for i in ci_ms.core_indices if i in frag_indices]
            A_actv = [
                [i for i in g if i in frag_indices] for g in ci_ms.active_orbitals
            ]
            A_virt = [i for i in ci_ms.virtual_indices if i in frag_indices]
            emb_space = EmbeddingMOSpace(
                nmo=aset.nmo,
                frozen_core_orbitals=list(aset.emb_space.frozen_core_orbitals),
                B_core_orbitals=list(aset_emb.B_core_orbitals),
                A_core_orbitals=A_core,
                active_orbitals=A_actv,
                A_virtual_orbitals=A_virt,
                B_virtual_orbitals=list(aset_emb.B_virtual_orbitals),
                frozen_virtual_orbitals=list(aset_emb.frozen_virtual_orbitals),
            )
        return aset, ci_driver, aset.system, full_mos, emb_space

    def _get_active_rdms(self, aset, ci_driver):
        """Return (g1, g2, l2, l3, nel_total) in the original active orbital basis."""
        if ci_driver is None:
            # Pure RHF or ROHF parent of ASET with no CI driver
            parent = aset.parent_method
            nel_total = parent.nel
            if isinstance(parent, forte2.scf.RHF):
                g1 = np.zeros((0, 0))
                g2 = np.zeros((0, 0, 0, 0))
                l2 = np.zeros((0, 0, 0, 0))
                l3 = None
                return g1, g2, l2, l3, nel_total
            raise NotImplementedError(
                "ASET2 on ROHF without a CI driver is not supported; wrap ROHF->ASET with CI first."
            )

        solver = ci_driver.ci_solver
        # Determine total electron count from the first state
        first_state = solver.sa_info.states[0]
        nel_total = first_state.nel

        if self.root is None:
            g1 = solver.make_average_rdm(1)
            g2 = solver.make_average_rdm(2)
            l2 = solver.make_average_cumulant(2)
            l3 = None if self.skip_3_cumulant else solver.make_average_cumulant(3)
        else:
            g1 = solver.make_rdm(self.root, order=1, spin_type="sf")
            g2 = solver.make_rdm(self.root, order=2, spin_type="sf")
            l2 = solver.make_cumulant(self.root, order=2, spin_type="sf")
            l3 = (
                None
                if self.skip_3_cumulant
                else solver.make_cumulant(self.root, order=3, spin_type="sf")
            )
        return g1, g2, l2, l3, nel_total

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        aset, ci_driver, full_system, full_mos, emb_space = self._resolve_upstream()
        g1_orig, g2_orig, l2_orig, l3_orig, nel_total = self._get_active_rdms(
            aset, ci_driver
        )

        # If ASET2 is attached directly to ASET (whose parent is MCOptimizer/CI),
        # ASET already rotated the active orbitals by aset.semican.Uactv without
        # updating the parent's CI vectors. Bring the parent RDMs into aset.mos's basis!
        if isinstance(self.parent_method, ASET) and g1_orig.shape[0] > 0:
            Ua0 = aset.semican.Uactv
            g1_orig = np.einsum("ip,ij,jq->pq", Ua0.conj(), g1_orig, Ua0, optimize=True)
            g2_orig = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                Ua0.conj(),
                Ua0.conj(),
                g2_orig,
                Ua0,
                Ua0,
                optimize=True,
            )
            l2_orig = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                Ua0.conj(),
                Ua0.conj(),
                l2_orig,
                Ua0,
                Ua0,
                optimize=True,
            )
            if l3_orig is not None:
                l3_orig = np.einsum(
                    "ip,jq,kr,ijklmn,ls,mt,nu->pqrstu",
                    Ua0.conj(),
                    Ua0.conj(),
                    Ua0.conj(),
                    l3_orig,
                    Ua0,
                    Ua0,
                    Ua0,
                    optimize=True,
                )

        # Semicanonicalize all 5 subspaces (and frozen spaces) with the chosen state's 1-RDM.
        # When attached directly to ASET with root=None, ASET already semicanonicalized
        # the exact same 5 subspaces with the exact same 1-RDM, so reuse aset.semican
        # directly (avoiding redundant Fock builds and arbitrary +/-1 eigenvector flips).
        if isinstance(self.parent_method, ASET) and self.root is None:
            semican = aset.semican
            C_semi = semican.C_semican
            fock_full = semican.fock_semican
            eps_full = semican.eps_semican
            Uactv = np.eye(g1_orig.shape[0])
        else:
            C_orig = full_mos.C[0]
            C_emb = C_orig[:, emb_space.orig_to_contig].copy()
            semican = Semicanonicalizer(
                system=full_system,
                mo_space=emb_space,
                mix_active=False,
                do_active=True,
            )
            semican.semi_canonicalize(g1=g1_orig, C_contig=C_emb)
            C_semi = semican.C_semican
            fock_full = semican.fock_semican
            eps_full = semican.eps_semican
            Uactv = semican.Uactv

        # Subspace sizes in contiguous embedding order:
        # [frozen_core | B_core | A_core | A_actv | A_virt | B_virt | frozen_virt]
        sl_frzc = emb_space.frozen_core
        sl_ob = emb_space.B_core
        sl_ca_full = emb_space.A_core
        sl_aa_full = emb_space.actv
        sl_va_full = emb_space.A_virt
        sl_vb = emb_space.B_virt

        n_frzc = sl_frzc.stop - sl_frzc.start
        nc = sl_ob.stop - sl_ob.start
        nc_a = sl_ca_full.stop - sl_ca_full.start
        na_a = sl_aa_full.stop - sl_aa_full.start
        nv_a = sl_va_full.stop - sl_va_full.start
        na = nc_a + na_a + nv_a
        nv = sl_vb.stop - sl_vb.start

        sl_A_full = slice(sl_ca_full.start, sl_va_full.stop)
        sl_corr_full = slice(sl_ob.start, sl_vb.stop)

        # Sub-slices inside fragment A = [C_A | A_A | V_A]
        sl_ca = slice(0, nc_a)
        sl_aa = slice(nc_a, nc_a + na_a)
        sl_va = slice(nc_a + na_a, na)

        # Rotate active-space RDMs/cumulants into the semicanonical A_A basis
        if na_a > 0:
            g1_a = np.einsum("ip,ij,jq->pq", Uactv, g1_orig, Uactv.conj(), optimize=True)
            g2_a = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                Uactv,
                Uactv,
                g2_orig,
                Uactv.conj(),
                Uactv.conj(),
                optimize=True,
            )
            l2_a = np.einsum(
                "ip,jq,ijkl,kr,ls->pqrs",
                Uactv,
                Uactv,
                l2_orig,
                Uactv.conj(),
                Uactv.conj(),
                optimize=True,
            )
            if l3_orig is not None:
                l3_a = np.einsum(
                    "ip,jq,kr,ijklmn,ls,mt,nu->pqrstu",
                    Uactv,
                    Uactv,
                    Uactv,
                    l3_orig,
                    Uactv.conj(),
                    Uactv.conj(),
                    Uactv.conj(),
                    optimize=True,
                )
            else:
                l3_a = None
        else:
            g1_a = np.zeros((0, 0))
            g2_a = np.zeros((0, 0, 0, 0))
            l2_a = np.zeros((0, 0, 0, 0))
            l3_a = None

        # Reference energy <Psi_0 | bare H^{A+B} | Psi_0>
        docc_all = list(range(0, sl_ca_full.stop))
        actv_all = list(range(sl_aa_full.start, sl_aa_full.stop))
        ints_ref = RestrictedMOIntegrals(
            system=full_system,
            C=C_semi,
            orbitals=actv_all,
            core_orbitals=docc_all,
        )
        self.E_ref = float(
            cas_energy_given_RDMs(ints_ref.E, ints_ref.H, ints_ref.V, g1_a, g2_a)
        )

        # Build fragment 1-RDM gamma1 and 1-hole density eta1 over A = {C_A, A_A, V_A}
        g1_A = np.zeros((na, na))
        g1_A[sl_ca, sl_ca] = 2.0 * np.eye(nc_a)
        if na_a > 0:
            g1_A[sl_aa, sl_aa] = g1_a
        e1_A = 2.0 * np.eye(na) - g1_A

        # Correlated outer DSRG space: core = O_B, actv = A, virt = V_B
        fock = fock_full[sl_corr_full, sl_corr_full].copy()
        eps = eps_full[sl_corr_full].copy()
        s_c = slice(0, nc)
        s_a = slice(nc, nc + na)
        s_v = slice(nc + na, nc + na + nv)
        s_h = slice(0, nc + na)
        s_p = slice(nc, nc + na + nv)
        hc = slice(0, nc)
        ha = slice(nc, nc + na)
        pa = slice(0, na)
        pv = slice(na, na + nv)

        eps_c = eps[s_c]
        eps_a = eps[s_a]
        eps_v = eps[s_v]
        eps_h = eps[s_h]
        eps_p = eps[s_p]

        # Build 3-center B blocks in semicanonical embedding basis
        fb = full_system.fock_builder
        C_ob = C_semi[:, sl_ob]
        C_A = C_semi[:, sl_A_full]
        C_vb = C_semi[:, sl_vb]

        B_ac = fb.B_tensor_gen_block(C_A, C_ob)  # (P, na, nc)
        B_vc = fb.B_tensor_gen_block(C_vb, C_ob)  # (P, nv, nc)
        B_aa = fb.B_tensor_gen_block(C_A, C_A)  # (P, na, na)
        B_va = fb.B_tensor_gen_block(C_vb, C_A)  # (P, nv, na)

        # Bare two-electron integral blocks needed for ASET(2)
        V = {
            "vvaa": np.einsum("Pau,Pbv->abuv", B_va, B_va, optimize=True),
            "aacc": np.einsum("Pui,Pvj->uvij", B_ac, B_ac, optimize=True),
            "avca": np.einsum("Pui,Pav->uaiv", B_ac, B_va, optimize=True),
            "avac": np.einsum("Puv,Pai->uavi", B_aa, B_vc, optimize=True),
            "vaaa": np.einsum("Pav,Pux->auvx", B_va, B_aa, optimize=True),
            "aaca": np.einsum("Pui,Pvx->uvix", B_ac, B_aa, optimize=True),
            "aaaa": np.einsum("Pux,Pvy->uvxy", B_aa, B_aa, optimize=True),
        }

        eps_dict = {"c": eps_c, "a": eps_a, "v": eps_v}
        s = self.flow_param

        # Build first-order T2 and S2 amplitudes
        T2 = {}
        for key in ("aavv", "ccaa", "caav", "acav", "aava", "caaa"):
            vkey = key[2:] + key[:2]
            T2[key] = V[vkey].transpose(2, 3, 0, 1).copy()
            compute_t2_block(T2[key], *(eps_dict[k] for k in key), s)

        S2 = {
            "aavv": 2.0 * T2["aavv"] - T2["aavv"].swapaxes(2, 3),
            "ccaa": 2.0 * T2["ccaa"] - T2["ccaa"].swapaxes(2, 3),
            "caav": 2.0 * T2["caav"] - T2["acav"].swapaxes(0, 1),
            "acav": 2.0 * T2["acav"] - T2["caav"].swapaxes(0, 1),
            "aava": 2.0 * T2["aava"] - T2["aava"].swapaxes(0, 1),
            "caaa": 2.0 * T2["caaa"] - T2["caaa"].swapaxes(2, 3),
        }

        # Diagonal H^(0) commutator with T2: Delta_u^x * gamma_u^x is nonzero only on A_A x A_A
        if na_a > 0:
            eps_aa = eps_a[sl_aa]
            dg_aa = 0.5 * (eps_aa[:, None] - eps_aa[None, :]) * g1_a
        else:
            dg_aa = np.zeros((0, 0))

        # Build first-order T1 amplitudes (Paper 2 Eq. 25)
        T1 = fock[s_h, s_p].copy()
        if na_a > 0:
            T1[hc, pa] += np.einsum(
                "ivau,uv->ia",
                S2["caaa"][:, sl_aa, :, sl_aa],
                dg_aa,
                optimize=True,
            )
            T1[hc, pv] += np.einsum(
                "vmue,uv->me",
                S2["acav"][sl_aa, :, sl_aa, :],
                dg_aa,
                optimize=True,
            )
            T1[ha, pv] += np.einsum(
                "ivau,uv->ia",
                S2["aava"][:, sl_aa, :, sl_aa],
                dg_aa,
                optimize=True,
            )
        compute_t1_block(T1, eps_h, eps_p, s)
        # Exclude internal fragment excitations
        T1[ha, pa] = 0.0
        if not self.include_env_corr:
            T1[hc, pv] = 0.0

        # Renormalized Fock matrix F_tilde (Paper 2 Eq. A1)
        F_tilde = fock[s_p, s_h].copy()
        if na_a > 0:
            F_tilde[pa, hc] += np.einsum(
                "ivau,uv->ai",
                S2["caaa"][:, sl_aa, :, sl_aa],
                dg_aa,
                optimize=True,
            )
            F_tilde[pv, hc] += np.einsum(
                "vmue,uv->em",
                S2["acav"][sl_aa, :, sl_aa, :],
                dg_aa,
                optimize=True,
            )
            F_tilde[pv, ha] += np.einsum(
                "ivau,uv->ai",
                S2["aava"][:, sl_aa, :, sl_aa],
                dg_aa,
                optimize=True,
            )
        delta_ph = eps_h[None, :] - eps_p[:, None]
        F_tilde *= np.exp(-s * delta_ph * delta_ph)
        F_tilde += fock[s_p, s_h]

        # Renormalize V blocks in place (leaving V["aaaa"] bare)
        for key in ("vvaa", "aacc", "avca", "avac", "vaaa", "aaca"):
            renormalize_V_block(V[key], *(eps_dict[k] for k in key), s)

        # ---- Second-order energies & streamed large-block C1 contributions ----
        # Pure-environment correlation delta E^B_{c,ASET(2)} (Paper 2 Eq. A3)
        delta_E_B = 0.0
        if self.include_env_corr and nc > 0 and nv > 0:
            delta_E_B += 2.0 * np.einsum(
                "em,me->", F_tilde[pv, hc], T1[hc, pv], optimize=True
            )
            delta_E_B += self._compute_ccvv_energy(B_vc, eps_c, eps_v, s)

        # Exact streamed cavv and ccav contributions to delta E^{AB} and C1
        E_cavv, C1_cavv = self._compute_cavv_exact(
            B_vc, B_va, eps_c, eps_a, eps_v, s, g1_A
        )
        E_ccav, C1_ccav = self._compute_ccav_exact(
            B_vc, B_ac, eps_c, eps_a, eps_v, s, e1_A
        )

        # Remaining fragment-environment energy delta E^{AB}_{c,ASET(2)} (Paper 2 Eq. A4)
        delta_E_AB = E_cavv + E_ccav
        delta_E_AB += self._compute_E_AB_nonstreamed(
            F_tilde, V, T1, T2, S2, g1_A, e1_A, l2_a, l3_a, hc, ha, pa, pv, sl_aa
        )

        self.delta_E_B = float(delta_E_B)
        self.delta_E_AB = float(delta_E_AB)
        self.E_aset2_ref = self.E_ref + self.delta_E_AB + self.delta_E_B

        # ---- Dressed effective Hamiltonian Hbar1, Hbar2 on fragment A ----
        C1 = C1_cavv - C1_ccav
        C1 += self._compute_C1_nonstreamed(
            F_tilde, V, T1, T2, S2, g1_A, e1_A, l2_a, hc, ha, pa, pv, sl_aa, na
        )
        hbar1_no = fock[s_a, s_a].copy() + 0.5 * (C1 + C1.T)

        C2 = self._compute_C2(F_tilde, V, T1, T2, S2, g1_A, e1_A, hc, ha, pa, pv, na)
        hbar2_no = V["aaaa"].copy() + 0.5 * (C2 + C2.transpose(2, 3, 0, 1))

        # De-normal-order Hbar^{[2]} w.r.t. |-_A> x |Phi_B> (Paper 2 Eqs. 28-30)
        hbar2_temp = 2.0 * hbar2_no - hbar2_no.swapaxes(2, 3)

        E_scalar = self.E_aset2_ref
        E_scalar -= np.einsum("vu,vu->", hbar1_no, g1_A, optimize=True)
        E_scalar += 0.25 * np.einsum(
            "uv,vyux,xy->", g1_A, hbar2_temp, g1_A, optimize=True
        )
        if na_a > 0:
            E_scalar -= 0.5 * np.einsum(
                "xyuv,uvxy->",
                hbar2_no[sl_aa, sl_aa, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )

        hbar1 = hbar1_no - 0.5 * np.einsum("uxvy,yx->uv", hbar2_temp, g1_A, optimize=True)
        hbar2 = hbar2_no

        self.E_scalar = float(E_scalar)
        self.hbar1_no = hbar1_no
        self.hbar2_no = hbar2_no
        self.hbar1 = hbar1
        self.hbar2 = hbar2

        # Package dressed fragment system, identity MOs, and fragment MOSpace
        self.nel_frag = nel_total - 2 * (n_frzc + nc)
        self.system = DressedFragmentSystem(
            scalar_energy=self.E_scalar,
            hcore=self.hbar1,
            V_phys=self.hbar2,
            nel_frag=self.nel_frag,
        )
        self.mos = MO(
            C=[np.eye(na)],
            spinorbital=False,
            irrep_indices=[[0] * na],
            irrep_labels=[["A"] * na],
        )

        # Preserve GAS structure of active orbitals inside fragment A
        actv_frag = []
        offset = nc_a
        for gas_sl in emb_space.gas:
            gsize = gas_sl.stop - gas_sl.start
            actv_frag.append(list(range(offset, offset + gsize)))
            offset += gsize

        self.mo_space = MOSpace(
            nmo=na,
            frozen_core_orbitals=[],
            core_orbitals=list(range(nc_a)),
            active_orbitals=actv_frag,
            frozen_virtual_orbitals=[],
        )

        logger.log_info1("\nASET(2) Downfolding Summary:")
        logger.log_info1(f"  Fragment orbitals (N_A): {na} (C_A={nc_a}, A_A={na_a}, V_A={nv_a})")
        logger.log_info1(f"  Environment orbitals:    O_B={nc}, V_B={nv}")
        logger.log_info1(f"  E_ref (bare):            {self.E_ref:20.12f} Eh")
        logger.log_info1(f"  delta E_AB:              {self.delta_E_AB:20.12f} Eh")
        logger.log_info1(f"  delta E_B:               {self.delta_E_B:20.12f} Eh")
        logger.log_info1(f"  E_ASET(2) (unrelaxed):   {self.E_aset2_ref:20.12f} Eh")

        self.executed = True
        return self

    # ------------------------------------------------------------------
    # Streamed large-block kernels (exact for u != v)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ccvv_energy(B_vc, eps_c, eps_v, s):
        nc = eps_c.shape[0]
        nv = eps_v.shape[0]
        if nc == 0 or nv == 0:
            return 0.0
        # cvB has shape (nc, nv, naux)
        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))
        E = 0.0
        for m in range(nc):
            # V_m[n, e, f] = (m e | n f) = <m n | e f>
            V_m = np.einsum("eP,nfP->nef", cvB[m], cvB, optimize=True)
            d = (
                eps_c[m]
                + eps_c[:, None, None]
                - eps_v[None, :, None]
                - eps_v[None, None, :]
            )
            # (1 - exp(-2 s d^2)) / d = (1 + exp(-s d^2)) * (1 - exp(-s d^2)) / d
            reg = _regularized_denom(d, s) * (1.0 + np.exp(-s * d * d))
            S_m = 2.0 * V_m - V_m.swapaxes(1, 2)
            E += float(np.einsum("nef,nef,nef->", V_m, reg, S_m, optimize=True))
        return E

    @staticmethod
    def _compute_cavv_exact(B_vc, B_va, eps_c, eps_a, eps_v, s, g1_A):
        nc = eps_c.shape[0]
        na = eps_a.shape[0]
        nv = eps_v.shape[0]
        C1_cavv = np.zeros((na, na))
        if nc == 0 or na == 0 or nv == 0:
            return 0.0, C1_cavv

        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))  # (nc, nv, P)
        vaB = np.ascontiguousarray(B_va.transpose(1, 2, 0))  # (nv, na, P)
        for m in range(nc):
            # V_m[e, f, u] = (m e | u f) = <m u | e f> = <e f | m u> = <f e | u m>
            V_m = np.einsum("eP,fuP->efu", cvB[m], vaB, optimize=True)
            d = (
                eps_c[m]
                + eps_a[None, None, :]
                - eps_v[:, None, None]
                - eps_v[None, :, None]
            )
            # Exact Vtilde_m[e, f, z] uses delta_{m z}^{e f}
            Vtilde_m = V_m * (1.0 + np.exp(-s * d * d))
            # Exact S2_m[e, f, w] uses delta_{m w}^{e f}
            T2_m = V_m * _regularized_denom(d, s)
            S2_m = 2.0 * T2_m - T2_m.swapaxes(0, 1)
            # Matches dsrg_common.py: C1[w, z] += sum_{ef} h2["vvac"][e, f, z, m] * s2["acvv"][w, m, e, f]
            # Since V_m[e, f, z] = <m z | e f> = <f e | z m> and S2_m[f, e, w] = S2[w, m, f, e]:
            # sum_{ef} Vtilde_m[e, f, z] * S2_m[e, f, w] == sum_{ef} h2["vvac"][e, f, z, m] * s2["acvv"][w, m, e, f]
            C1_cavv += np.einsum("efz,efw->wz", Vtilde_m, S2_m, optimize=True)

        # Energy: E += sum_{wz} C1_cavv[w, z] * g1_A[z, w]
        E_cavv = float(np.einsum("wz,zw->", C1_cavv, g1_A, optimize=True))
        return E_cavv, C1_cavv

    @staticmethod
    def _compute_ccav_exact(B_vc, B_ac, eps_c, eps_a, eps_v, s, e1_A):
        nc = eps_c.shape[0]
        na = eps_a.shape[0]
        nv = eps_v.shape[0]
        C1_ccav = np.zeros((na, na))
        if nc == 0 or na == 0 or nv == 0:
            return 0.0, C1_ccav

        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))  # (nc, nv, P)
        caB = np.ascontiguousarray(B_ac.transpose(2, 1, 0))  # (nc, na, P)
        for m in range(nc):
            # V_eu[n, e, u] = (m e | n u) = <m n | e u> = <e u | m n>
            V_eu = np.einsum("eP,nuP->neu", cvB[m], caB, optimize=True)
            # V_ue[n, e, u] = (m u | n e) = <m n | u e> = <n m | e u>
            V_ue = np.einsum("uP,neP->neu", caB[m], cvB, optimize=True)
            d = (
                eps_c[m]
                + eps_c[:, None, None]
                - eps_v[None, :, None]
                - eps_a[None, None, :]
            )
            # Matches dsrg_common.py:
            # C1[w, z] -= sum_{n,m,e} h2["vacc"][e, w, n, m] * s2["ccva"][n, m, e, z]
            # Note h2["vacc"][e, w, m, n] = Vtilde_eu[n, e, w], s2["ccva"][m, n, e, z] = (2*V_eu - V_ue) * denom
            Vtilde_m = V_eu * (1.0 + np.exp(-s * d * d))
            S2_m = (2.0 * V_eu - V_ue) * _regularized_denom(d, s)
            C1_ccav += np.einsum("new,nez->wz", Vtilde_m, S2_m, optimize=True)

        # Energy: E += sum_{wz} C1_ccav[w, z] * e1_A[z, w]
        E_ccav = float(np.einsum("wz,zw->", C1_ccav, e1_A, optimize=True))
        return E_ccav, C1_ccav

    # ------------------------------------------------------------------
    # Non-streamed contractions (block-sparse in lambda2 and lambda3!)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_E_AB_nonstreamed(
        h1, h2, t1, t2, s2, g1, e1, l2_a, l3_a, hc, ha, pa, pv, sl_aa
    ):
        E = 0.0
        # H1_T1_C0 (excluding pure-environment 2 * h1[pv, hc] * t1[hc, pv] which is in delta_E_B)
        E += 2.0 * np.einsum("um,mu->", h1[pa, hc], t1[hc, pa], optimize=True)
        E += np.einsum("ev,ue,uv->", h1[pv, ha], t1[ha, pv], g1, optimize=True)
        E -= np.einsum("um,mv,uv->", h1[pa, hc], t1[hc, pa], g1, optimize=True)

        # H2_T2_C0 (1-RDM / 1-hole density terms)
        E += 0.25 * np.einsum("efxu,yvef,uv,xy->", h2["vvaa"], s2["aavv"], g1, g1, optimize=True)
        E += 0.25 * np.einsum("vymn,mnux,uv,xy->", h2["aacc"], s2["ccaa"], e1, e1, optimize=True)
        E += 0.5 * np.einsum("vemx,myue,uv,xy->", h2["avca"], s2["caav"], e1, g1, optimize=True)
        E += 0.5 * np.einsum("vexm,ymue,uv,xy->", h2["avac"], s2["acav"], e1, g1, optimize=True)
        E += 0.25 * np.einsum(
            "evwx,zyeu,wz,uv,xy->", h2["vaaa"], s2["aava"], g1, e1, g1, optimize=True
        )
        E += 0.25 * np.einsum(
            "vzmx,myuw,wz,uv,xy->", h2["aaca"], s2["caaa"], e1, e1, g1, optimize=True
        )

        # Lambda2 terms (nonzero ONLY on A_A^4 = sl_aa!)
        if l2_a.size > 0:
            h1_pv_aa = h1[pv, ha][:, sl_aa]
            h1_aa_hc = h1[pa, hc][sl_aa, :]
            t1_aa_pv = t1[ha, pv][sl_aa, :]
            t1_hc_aa = t1[hc, pa][:, sl_aa]

            E += np.einsum(
                "ex,uvey,uvxy->",
                h1_pv_aa,
                t2["aava"][sl_aa, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            E -= np.einsum(
                "vm,muyx,uvxy->",
                h1_aa_hc,
                t2["caaa"][:, sl_aa, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )
            E += np.einsum(
                "evxy,ue,uvxy->",
                h2["vaaa"][:, sl_aa, sl_aa, sl_aa],
                t1_aa_pv,
                l2_a,
                optimize=True,
            )
            E -= np.einsum(
                "uvmy,mx,uvxy->",
                h2["aaca"][sl_aa, sl_aa, :, sl_aa],
                t1_hc_aa,
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "uvmn,mnxy,uvxy->",
                h2["aacc"][sl_aa, sl_aa, :, :],
                t2["ccaa"][:, :, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "uvmw,mzxy,wz,uvxy->",
                h2["aaca"][sl_aa, sl_aa, :, :],
                t2["caaa"][:, :, sl_aa, sl_aa],
                g1,
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "efxy,uvef,uvxy->",
                h2["vvaa"][:, :, sl_aa, sl_aa],
                t2["aavv"][sl_aa, sl_aa, :, :],
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "ezxy,uvew,wz,uvxy->",
                h2["vaaa"][:, :, sl_aa, sl_aa],
                t2["aava"][sl_aa, sl_aa, :, :],
                e1,
                l2_a,
                optimize=True,
            )
            E += np.einsum(
                "uexm,vmye,uvxy->",
                h2["avac"][sl_aa, :, sl_aa, :],
                s2["acav"][sl_aa, :, sl_aa, :],
                l2_a,
                optimize=True,
            )
            E -= np.einsum(
                "uemx,vmye,uvxy->",
                h2["avca"][sl_aa, :, :, sl_aa],
                t2["acav"][sl_aa, :, sl_aa, :],
                l2_a,
                optimize=True,
            )
            E -= np.einsum(
                "vemx,muye,uvxy->",
                h2["avca"][sl_aa, :, :, sl_aa],
                t2["caav"][:, sl_aa, sl_aa, :],
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "euwx,zvey,wz,uvxy->",
                h2["vaaa"][:, sl_aa, :, sl_aa],
                s2["aava"][:, sl_aa, :, sl_aa],
                g1,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "euxw,zvey,wz,uvxy->",
                h2["vaaa"][:, sl_aa, sl_aa, :],
                t2["aava"][:, sl_aa, :, sl_aa],
                g1,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "evxw,uzey,wz,uvxy->",
                h2["vaaa"][:, sl_aa, sl_aa, :],
                t2["aava"][sl_aa, :, :, sl_aa],
                g1,
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "wumx,mvzy,wz,uvxy->",
                h2["aaca"][:, sl_aa, :, sl_aa],
                s2["caaa"][:, sl_aa, :, sl_aa],
                e1,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "uwmx,mvzy,wz,uvxy->",
                h2["aaca"][sl_aa, :, :, sl_aa],
                t2["caaa"][:, sl_aa, :, sl_aa],
                e1,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "vwmx,muyz,wz,uvxy->",
                h2["aaca"][sl_aa, :, :, sl_aa],
                t2["caaa"][:, sl_aa, sl_aa, :],
                e1,
                l2_a,
                optimize=True,
            )

        # Lambda3 terms (Paper 2 Eq. 35; nonzero ONLY on A_A^6!)
        if l3_a is not None and l3_a.size > 0:
            E += np.einsum(
                "ewxy,uvez,xyzuwv->",
                h2["vaaa"][:, sl_aa, sl_aa, sl_aa],
                t2["aava"][sl_aa, sl_aa, :, sl_aa],
                l3_a,
                optimize=True,
            )
            E -= np.einsum(
                "uvmz,mwxy,xyzuwv->",
                h2["aaca"][sl_aa, sl_aa, :, sl_aa],
                t2["caaa"][:, sl_aa, sl_aa, sl_aa],
                l3_a,
                optimize=True,
            )

        return float(E)

    @staticmethod
    def _compute_C1_nonstreamed(
        h1, h2, t1, t2, s2, g1, e1, l2_a, hc, ha, pa, pv, sl_aa, na
    ):
        C1 = np.zeros((na, na))

        # Singles & F_tilde contractions
        C1 += np.einsum("ev,ue->uv", h1[pv, ha], t1[ha, pv], optimize=True)
        C1 -= np.einsum("um,mv->uv", h1[pa, hc], t1[hc, pa], optimize=True)
        C1 += np.einsum("em,umve->uv", h1[pv, hc], s2["acav"], optimize=True)
        C1 += np.einsum("xm,muxv->uv", h1[pa, hc], s2["caaa"], optimize=True)
        C1 += 0.50 * np.einsum("ex,yuev,xy->uv", h1[pv, ha], s2["aava"], g1, optimize=True)
        C1 -= 0.50 * np.einsum("ym,muxv,xy->uv", h1[pa, hc], s2["caaa"], g1, optimize=True)

        # Pure two-body contractions (no density matrix)
        C1 += np.einsum("uemz,mwue->wz", h2["avca"], s2["caav"], optimize=True)
        C1 += np.einsum("uezm,wmue->wz", h2["avac"], s2["acav"], optimize=True)
        C1 += np.einsum("vumz,mwvu->wz", h2["aaca"], s2["caaa"], optimize=True)
        C1 -= np.einsum("wemu,muze->wz", h2["avca"], s2["caav"], optimize=True)
        C1 -= np.einsum("weum,umze->wz", h2["avac"], s2["acav"], optimize=True)
        C1 -= np.einsum("ewvu,vuez->wz", h2["vaaa"], s2["aava"], optimize=True)

        # Contractions with g1 (avoiding any N_A^4 intermediate)
        C1 += 0.5 * np.einsum("wvef,efzu,uv->wz", s2["aavv"], h2["vvaa"], g1, optimize=True)
        C1 += 0.5 * np.einsum("wvex,exzu,uv->wz", s2["aava"], h2["vaaa"], g1, optimize=True)
        C1 += 0.5 * np.einsum("vwex,exuz,uv->wz", s2["aava"], h2["vaaa"], g1, optimize=True)
        C1 -= 0.5 * np.einsum("wmue,vezm,uv->wz", s2["acav"], h2["avac"], g1, optimize=True)
        C1 -= 0.5 * np.einsum("mwxu,xvmz,uv->wz", s2["caaa"], h2["aaca"], g1, optimize=True)
        C1 -= 0.5 * np.einsum("mwue,vemz,uv->wz", s2["caav"], h2["avca"], g1, optimize=True)
        C1 -= 0.5 * np.einsum("mwux,vxmz,uv->wz", s2["caaa"], h2["aaca"], g1, optimize=True)
        C1 += 0.25 * np.einsum(
            "jwxu,xy,yvjz,uv->wz", s2["caaa"], g1, h2["aaca"], g1, optimize=True
        )
        C1 -= 0.25 * np.einsum(
            "ywbu,xy,bvxz,uv->wz", s2["aava"], g1, h2["vaaa"], g1, optimize=True
        )
        C1 -= 0.25 * np.einsum(
            "wybu,xy,bvzx,uv->wz", s2["aava"], g1, h2["vaaa"], g1, optimize=True
        )

        # Contractions with e1 (avoiding any N_A^4 intermediate)
        C1 -= 0.5 * np.einsum("mnzu,wvmn,uv->wz", s2["ccaa"], h2["aacc"], e1, optimize=True)
        C1 -= 0.5 * np.einsum("mxzu,wvmx,uv->wz", s2["caaa"], h2["aaca"], e1, optimize=True)
        C1 -= 0.5 * np.einsum("mxuz,vwmx,uv->wz", s2["caaa"], h2["aaca"], e1, optimize=True)
        C1 += 0.5 * np.einsum("vmze,weum,uv->wz", s2["acav"], h2["avac"], e1, optimize=True)
        C1 += 0.5 * np.einsum("xvez,ewxu,uv->wz", s2["aava"], h2["vaaa"], e1, optimize=True)
        C1 += 0.5 * np.einsum("mvze,wemu,uv->wz", s2["caav"], h2["avca"], e1, optimize=True)
        C1 += 0.5 * np.einsum("vxez,ewux,uv->wz", s2["aava"], h2["vaaa"], e1, optimize=True)
        C1 -= 0.25 * np.einsum(
            "yvbz,xy,bwxu,uv->wz", s2["aava"], e1, h2["vaaa"], e1, optimize=True
        )
        C1 += 0.25 * np.einsum(
            "jvxz,xy,ywju,uv->wz", s2["caaa"], e1, h2["aaca"], e1, optimize=True
        )
        C1 += 0.25 * np.einsum(
            "jvzx,xy,wyju,uv->wz", s2["caaa"], e1, h2["aaca"], e1, optimize=True
        )

        # G2 intermediates for T1 and lambda2 contractions
        G2_avac = 2.0 * h2["avac"] - h2["avca"].transpose(0, 1, 3, 2)
        G2_aaac = 2.0 * h2["aaca"].transpose(1, 0, 3, 2) - h2["aaca"].transpose(0, 1, 3, 2)
        G2_avaa = 2.0 * h2["vaaa"].transpose(1, 0, 3, 2) - h2["vaaa"].transpose(1, 0, 2, 3)

        C1 += np.einsum("ma,uavm->uv", t1[hc, pa], G2_aaac, optimize=True)
        C1 += np.einsum("ma,uavm->uv", t1[hc, pv], G2_avac, optimize=True)
        C1 += 0.50 * np.einsum("xe,yx,uevy->uv", t1[ha, pv], g1, G2_avaa, optimize=True)
        C1 -= 0.50 * np.einsum("mx,xy,uyvm->uv", t1[hc, pa], g1, G2_aaac, optimize=True)

        # Lambda2 terms (nonzero ONLY on A_A^4 = sl_aa!)
        if l2_a.size > 0:
            C1 += 0.50 * np.einsum(
                "vujz,jwyx,xyuv->wz",
                h2["aaca"][sl_aa, sl_aa, :, :],
                t2["caaa"][:, :, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 += 0.50 * np.einsum(
                "auzx,wvay,xyuv->wz",
                h2["vaaa"][:, sl_aa, :, sl_aa],
                s2["aava"][:, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "auxz,wvay,xyuv->wz",
                h2["vaaa"][:, sl_aa, sl_aa, :],
                t2["aava"][:, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "auxz,vway,xyvu->wz",
                h2["vaaa"][:, sl_aa, sl_aa, :],
                t2["aava"][sl_aa, :, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "bwyx,vubz,xyuv->wz",
                h2["vaaa"][:, :, sl_aa, sl_aa],
                t2["aava"][sl_aa, sl_aa, :, :],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "wuix,ivzy,xyuv->wz",
                h2["aaca"][:, sl_aa, :, sl_aa],
                s2["caaa"][:, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 += 0.50 * np.einsum(
                "uwix,ivzy,xyuv->wz",
                h2["aaca"][sl_aa, :, :, sl_aa],
                t2["caaa"][:, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 += 0.50 * np.einsum(
                "uwix,ivyz,xyvu->wz",
                h2["aaca"][sl_aa, :, :, sl_aa],
                t2["caaa"][:, sl_aa, sl_aa, :],
                l2_a,
                optimize=True,
            )
            C1 += 0.50 * np.einsum(
                "avxy,uwaz,xyuv->wz",
                h2["vaaa"][:, sl_aa, sl_aa, sl_aa],
                s2["aava"][sl_aa, :, :, :],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "uviy,iwxz,xyuv->wz",
                h2["aaca"][sl_aa, sl_aa, :, sl_aa],
                s2["caaa"][:, :, sl_aa, :],
                l2_a,
                optimize=True,
            )
            C1 += 0.50 * np.einsum(
                "wezx,uvey,xyuv->wz",
                G2_avaa[:, :, :, sl_aa],
                t2["aava"][sl_aa, sl_aa, :, sl_aa],
                l2_a,
                optimize=True,
            )
            C1 -= 0.50 * np.einsum(
                "wuzm,mvxy,xyuv->wz",
                G2_aaac[:, sl_aa, :, :],
                t2["caaa"][:, sl_aa, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )

        return C1

    @staticmethod
    def _compute_C2(h1, h2, t1, t2, s2, g1, e1, hc, ha, pa, pv, na):
        C2 = np.zeros((na, na, na, na))
        C2 += np.einsum("efxy,uvef->uvxy", h2["vvaa"], t2["aavv"], optimize=True)
        C2 += np.einsum("ewxy,uvew->uvxy", h2["vaaa"], t2["aava"], optimize=True)
        C2 += np.einsum("ewyx,vuew->uvxy", h2["vaaa"], t2["aava"], optimize=True)

        C2 += np.einsum("uvmn,mnxy->uvxy", h2["aacc"], t2["ccaa"], optimize=True)
        C2 += np.einsum("vumw,mwyx->uvxy", h2["aaca"], t2["caaa"], optimize=True)
        C2 += np.einsum("uvmw,mwxy->uvxy", h2["aaca"], t2["caaa"], optimize=True)

        temp = np.einsum("ax,uvay->uvxy", h1[pv, ha], t2["aava"], optimize=True)
        temp -= np.einsum("ui,ivxy->uvxy", h1[pa, hc], t2["caaa"], optimize=True)
        temp += np.einsum("ua,avxy->uvxy", t1[ha, pv], h2["vaaa"], optimize=True)
        temp -= np.einsum("ix,uviy->uvxy", t1[hc, pa], h2["aaca"], optimize=True)

        temp -= 0.50 * np.einsum(
            "wz,vuaw,azyx->uvxy", g1, t2["aava"], h2["vaaa"], optimize=True
        )
        temp -= 0.50 * np.einsum(
            "wz,izyx,vuiw->uvxy", e1, t2["caaa"], h2["aaca"], optimize=True
        )

        temp += np.einsum("uexm,vmye->uvxy", h2["avac"], s2["acav"], optimize=True)
        temp += np.einsum("wumx,mvwy->uvxy", h2["aaca"], s2["caaa"], optimize=True)

        temp += 0.50 * np.einsum(
            "wz,zvay,auwx->uvxy", g1, s2["aava"], h2["vaaa"], optimize=True
        )
        temp -= 0.50 * np.einsum(
            "wz,ivwy,zuix->uvxy", g1, s2["caaa"], h2["aaca"], optimize=True
        )

        temp -= np.einsum("uemx,vmye->uvxy", h2["avca"], t2["acav"], optimize=True)
        temp -= np.einsum("uwmx,mvwy->uvxy", h2["aaca"], t2["caaa"], optimize=True)

        temp -= 0.50 * np.einsum(
            "wz,zvay,auxw->uvxy", g1, t2["aava"], h2["vaaa"], optimize=True
        )
        temp += 0.50 * np.einsum(
            "wz,ivwy,uzix->uvxy", g1, t2["caaa"], h2["aaca"], optimize=True
        )

        temp -= np.einsum("vemx,muye->uvxy", h2["avca"], t2["caav"], optimize=True)
        temp -= np.einsum("vwmx,muyw->uvxy", h2["aaca"], t2["caaa"], optimize=True)

        temp -= 0.50 * np.einsum(
            "wz,uzay,avxw->uvxy", g1, t2["aava"], h2["vaaa"], optimize=True
        )
        temp += 0.50 * np.einsum(
            "wz,iuyw,vzix->uvxy", g1, t2["caaa"], h2["aaca"], optimize=True
        )

        C2 += temp
        C2 += temp.transpose(1, 0, 3, 2)
        return C2
