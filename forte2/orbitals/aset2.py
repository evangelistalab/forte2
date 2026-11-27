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
    if s == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    thresh = 1e-4 / np.sqrt(s)
    small = np.abs(x) <= thresh
    if not np.any(small):
        return (1.0 - np.exp(-s * (x * x))) / x
    x_safe = np.where(small, 1.0, x)
    out = (1.0 - np.exp(-s * (x_safe * x_safe))) / x_safe
    xs = x[small]
    z2 = s * (xs * xs)
    # (1 - exp(-s * x^2)) / x = s * x * (1 - z^2/2 + z^4/6 - z^6/24 + z^8/120)
    out[small] = (
        s * xs * (1.0 - z2 / 2.0 + z2 * z2 / 6.0 - (z2**3) / 24.0 + (z2**4) / 120.0)
    )
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
        self._register_parent_method(parent_method)
        if not (
            isinstance(parent_method, ASET)
            or (
                isinstance(parent_method, ActiveSpaceDriver)
                and isinstance(parent_method.parent_method, ASET)
            )
        ):
            raise TypeError(
                "ASET2 parent must be ASET or a CI/MCOptimizer driver whose parent is ASET, "
                f"got {type(parent_method).__name__}."
            )
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
                frozen_core_orbitals=list(aset_emb.frozen_core_orbitals),
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
            # Pure RHF parent of ASET with no CI driver
            parent = aset.parent_method
            if isinstance(parent, forte2.scf.RHF):
                g1 = np.zeros((0, 0))
                g2 = np.zeros((0, 0, 0, 0))
                l2 = np.zeros((0, 0, 0, 0))
                l3 = None
                return g1, g2, l2, l3, parent.nel
            raise NotImplementedError(
                f"ASET2 on {type(parent).__name__} without a CI driver is not supported; "
                "wrap ASET with CI or MCOptimizer first."
            )

        solver = ci_driver.ci_solver
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
        sl_ha = slice(0, nc_a + na_a)
        sl_pa = slice(nc_a, na)

        # Rotate active-space RDMs/cumulants into the semicanonical A_A basis
        if na_a > 0:
            g1_a = np.einsum(
                "ip,ij,jq->pq", Uactv, g1_orig, Uactv.conj(), optimize=True
            )
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

        # Reference energy <Psi_0 | bare H^{A+B} | Psi_0>.
        # Reuse upstream CI/MCOptimizer/RHF energy directly to avoid a redundant
        # full-AO RestrictedMOIntegrals pass.
        if ci_driver is not None:
            self.E_ref = float(
                ci_driver.E if self.root is None else ci_driver.E_ci[self.root]
            )
        elif isinstance(aset.parent_method, forte2.scf.RHF):
            self.E_ref = float(aset.parent_method.E)
        else:
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
        g1_ha = np.ascontiguousarray(g1_A[sl_ha, sl_ha])
        e1_pa = np.ascontiguousarray(e1_A[sl_pa, sl_pa])

        # Correlated outer DSRG space: core = O_B, actv = A, virt = V_B
        fock = fock_full[sl_corr_full, sl_corr_full].copy()
        eps = np.diag(fock).copy()
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

        # Build 3-center B blocks in semicanonical embedding basis.
        # For real DF systems, perform a single L3-chunked half-transform scan of
        # B_Pmn against [C_A, C_ob] followed by a shared 2D BLAS DGEMM against
        # [C_A, C_vb], avoiding 4 full scans of B_Pmn and O(N_aux*N_bf^2*nv) work.
        fb = full_system.fock_builder
        C_ob = C_semi[:, sl_ob]
        C_A = C_semi[:, sl_A_full]
        C_vb = C_semi[:, sl_vb]

        if hasattr(fb, "B_Pmn") and np.isrealobj(C_semi) and np.isrealobj(fb.B_Pmn):
            B_Pmn = fb.B_Pmn
            naux_df, nbf, _ = B_Pmn.shape
            n_col = na + nc
            n_row = na + nv
            B_aa = np.empty((naux_df, na, na), dtype=B_Pmn.dtype)
            B_ac = np.empty((naux_df, na, nc), dtype=B_Pmn.dtype)
            B_va = np.empty((naux_df, nv, na), dtype=B_Pmn.dtype)
            B_vc = np.empty((naux_df, nv, nc), dtype=B_Pmn.dtype)
            if n_col > 0 and n_row > 0:
                C_A_ob = np.ascontiguousarray(np.hstack([C_A, C_ob]))
                C_A_vb = np.ascontiguousarray(np.hstack([C_A, C_vb]))
                P_chunk = max(1, min(naux_df, 256))
                for p0 in range(0, naux_df, P_chunk):
                    p1 = min(naux_df, p0 + P_chunk)
                    np_blk = p1 - p0
                    T_blk = (B_Pmn[p0:p1].reshape(np_blk * nbf, nbf) @ C_A_ob).reshape(
                        np_blk, nbf, n_col
                    )
                    T_blk_t = np.ascontiguousarray(T_blk.transpose(0, 2, 1)).reshape(
                        np_blk * n_col, nbf
                    )
                    Out = (T_blk_t @ C_A_vb).reshape(np_blk, n_col, n_row)
                    B_aa[p0:p1] = Out[:, :na, :na].transpose(0, 2, 1)
                    B_ac[p0:p1] = Out[:, na:, :na].transpose(0, 2, 1)
                    B_va[p0:p1] = Out[:, :na, na:].transpose(0, 2, 1)
                    B_vc[p0:p1] = Out[:, na:, na:].transpose(0, 2, 1)
        else:
            B_ac = fb.B_tensor_gen_block(C_A, C_ob)  # (P, na, nc)
            B_vc = fb.B_tensor_gen_block(C_vb, C_ob)  # (P, nv, nc)
            B_aa = fb.B_tensor_gen_block(C_A, C_A)  # (P, na, na)
            B_va = fb.B_tensor_gen_block(C_vb, C_A)  # (P, nv, na)

        # Bare two-electron integral blocks via BLAS 2D DGEMM (vvaa is streamed!)
        naux = B_aa.shape[0]
        B_ac_2d = B_ac.reshape(naux, na * nc)
        B_vc_2d = B_vc.reshape(naux, nv * nc)
        B_aa_2d = B_aa.reshape(naux, na * na)
        B_va_2d = B_va.reshape(naux, nv * na)
        V = {
            "aacc": np.ascontiguousarray(
                (B_ac_2d.T @ B_ac_2d).reshape(na, nc, na, nc).transpose(0, 2, 1, 3)
            ),
            "avca": np.ascontiguousarray(
                (B_ac_2d.T @ B_va_2d).reshape(na, nc, nv, na).transpose(0, 2, 1, 3)
            ),
            "avac": np.ascontiguousarray(
                (B_aa_2d.T @ B_vc_2d).reshape(na, na, nv, nc).transpose(0, 2, 1, 3)
            ),
            "vaaa": np.ascontiguousarray(
                (B_va_2d.T @ B_aa_2d).reshape(nv, na, na, na).transpose(0, 2, 1, 3)
            ),
            "aaca": np.ascontiguousarray(
                (B_ac_2d.T @ B_aa_2d).reshape(na, nc, na, na).transpose(0, 2, 1, 3)
            ),
            "aaaa": np.ascontiguousarray(
                (B_aa_2d.T @ B_aa_2d).reshape(na, na, na, na).transpose(0, 2, 1, 3)
            ),
        }

        eps_dict = {"c": eps_c, "a": eps_a, "v": eps_v}
        s = self.flow_param

        # Build first-order T2 and S2 amplitudes (aavv is streamed in _compute_vvaa_exact)
        T2 = {}
        for key in ("ccaa", "caav", "acav", "aava", "caaa"):
            vkey = key[2:] + key[:2]
            T2[key] = V[vkey].transpose(2, 3, 0, 1).copy()
            compute_t2_block(T2[key], *(eps_dict[k] for k in key), s)

        S2 = {
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
        for key in ("aacc", "avca", "avac", "vaaa", "aaca"):
            renormalize_V_block(V[key], *(eps_dict[k] for k in key), s)

        # ---- Second-order energies & streamed large-block C1/C2 contributions ----
        # Pure-environment correlation delta E^B_{c,ASET(2)} (Paper 2 Eq. A3)
        delta_E_B = 0.0
        if self.include_env_corr and nc > 0 and nv > 0:
            delta_E_B += 2.0 * np.einsum(
                "em,me->", F_tilde[pv, hc], T1[hc, pv], optimize=True
            )
            delta_E_B += self._compute_ccvv_energy(B_vc, eps_c, eps_v, s)

        # Exact streamed cavv, ccav, and vvaa contributions
        E_cavv, C1_cavv = self._compute_cavv_exact(
            B_vc, B_va, eps_c, eps_a, eps_v, s, g1_ha, sl_ha
        )
        E_ccav, C1_ccav = self._compute_ccav_exact(
            B_vc, B_ac, eps_c, eps_a, eps_v, s, e1_pa, sl_pa
        )
        E_vvaa, C1_vvaa, C2_vvaa = self._compute_vvaa_exact(
            B_va, eps_a, eps_v, s, g1_ha, l2_a, sl_ha, sl_aa
        )

        # Remaining fragment-environment energy delta E^{AB}_{c,ASET(2)} (Paper 2 Eq. A4)
        delta_E_AB = E_cavv + E_ccav + E_vvaa
        delta_E_AB += self._compute_E_AB_nonstreamed(
            F_tilde,
            V,
            T1,
            T2,
            S2,
            g1_ha,
            e1_pa,
            l2_a,
            l3_a,
            hc,
            ha,
            pa,
            pv,
            sl_ha,
            sl_pa,
            sl_aa,
        )

        self.delta_E_B = float(delta_E_B)
        self.delta_E_AB = float(delta_E_AB)
        self.E_aset2_ref = self.E_ref + self.delta_E_AB + self.delta_E_B

        # ---- Dressed effective Hamiltonian Hbar1, Hbar2 on fragment A ----
        C1 = C1_cavv - C1_ccav + C1_vvaa
        C1 += self._compute_C1_nonstreamed(
            F_tilde,
            V,
            T1,
            T2,
            S2,
            g1_ha,
            e1_pa,
            l2_a,
            hc,
            ha,
            pa,
            pv,
            sl_ha,
            sl_pa,
            sl_aa,
            na,
        )
        hbar1_no = fock[s_a, s_a].copy() + 0.5 * (C1 + C1.T)

        C2 = self._compute_C2(
            F_tilde, V, T1, T2, S2, g1_ha, sl_ha, hc, ha, pa, pv, na, C2_vvaa
        )
        hbar2_no = V["aaaa"].copy() + 0.5 * (C2 + C2.transpose(2, 3, 0, 1))

        # De-normal-order Hbar^{[2]} w.r.t. |-_A> x |Phi_B> (Paper 2 Eqs. 28-30)
        hbar2_temp = 2.0 * hbar2_no - hbar2_no.swapaxes(2, 3)

        E_scalar = self.E_aset2_ref
        E_scalar -= np.einsum("vu,vu->", hbar1_no[sl_ha, sl_ha], g1_ha, optimize=True)
        E_scalar += 0.25 * np.einsum(
            "uv,vyux,xy->",
            g1_ha,
            hbar2_temp[sl_ha, sl_ha, sl_ha, sl_ha],
            g1_ha,
            optimize=True,
        )
        if na_a > 0:
            E_scalar -= 0.5 * np.einsum(
                "xyuv,uvxy->",
                hbar2_no[sl_aa, sl_aa, sl_aa, sl_aa],
                l2_a,
                optimize=True,
            )

        hbar1 = hbar1_no - 0.5 * np.einsum(
            "uxvy,yx->uv", hbar2_temp[:, sl_ha, :, sl_ha], g1_ha, optimize=True
        )
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
        logger.log_info1(
            f"  Fragment orbitals (N_A): {na} (C_A={nc_a}, A_A={na_a}, V_A={nv_a})"
        )
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
        naux = B_vc.shape[0]
        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))  # (nc, nv, P)
        cvB_2d = cvB.reshape(nc * nv, naux)
        E = 0.0
        for m in range(nc):
            # V_m[n, e, f] = (m e | n f) = <m n | e f>
            V_m = (cvB[m] @ cvB_2d.T).reshape(nv, nc, nv).transpose(1, 0, 2)
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
    def _compute_cavv_exact(B_vc, B_va, eps_c, eps_a, eps_v, s, g1_ha, sl_ha):
        nc = eps_c.shape[0]
        na = eps_a.shape[0]
        nv = eps_v.shape[0]
        C1_cavv = np.zeros((na, na))
        if nc == 0 or na == 0 or nv == 0:
            return 0.0, C1_cavv

        naux = B_vc.shape[0]
        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))  # (nc, nv, P)
        vaB = np.ascontiguousarray(B_va.transpose(1, 2, 0))  # (nv, na, P)
        vaB_2d = vaB.reshape(nv * na, naux)
        for m in range(nc):
            # V_m[e, f, u] = (m e | u f) = <m u | e f> = <e f | m u> = <f e | u m>
            V_m = (cvB[m] @ vaB_2d.T).reshape(nv, nv, na)
            d = (
                eps_c[m]
                + eps_a[None, None, :]
                - eps_v[:, None, None]
                - eps_v[None, :, None]
            )
            Vtilde_m = np.ascontiguousarray(V_m * (1.0 + np.exp(-s * d * d)))
            T2_m = V_m * _regularized_denom(d, s)
            S2_m = np.ascontiguousarray(2.0 * T2_m - T2_m.swapaxes(0, 1))
            # C1_cavv[w, z] += sum_{ef} S2_m[e, f, w] * Vtilde_m[e, f, z]
            C1_cavv += S2_m.reshape(nv * nv, na).T @ Vtilde_m.reshape(nv * nv, na)

        # Energy: E += sum_{wz in sl_ha} C1_cavv[w, z] * g1_ha[z, w]
        E_cavv = float(
            np.einsum("wz,zw->", C1_cavv[sl_ha, sl_ha], g1_ha, optimize=True)
        )
        return E_cavv, C1_cavv

    @staticmethod
    def _compute_ccav_exact(B_vc, B_ac, eps_c, eps_a, eps_v, s, e1_pa, sl_pa):
        nc = eps_c.shape[0]
        na = eps_a.shape[0]
        nv = eps_v.shape[0]
        C1_ccav = np.zeros((na, na))
        if nc == 0 or na == 0 or nv == 0:
            return 0.0, C1_ccav

        naux = B_vc.shape[0]
        cvB = np.ascontiguousarray(B_vc.transpose(2, 1, 0))  # (nc, nv, P)
        caB = np.ascontiguousarray(B_ac.transpose(2, 1, 0))  # (nc, na, P)
        cvB_2d = cvB.reshape(nc * nv, naux)
        caB_2d = caB.reshape(nc * na, naux)
        for m in range(nc):
            # V_eu[n, e, u] = (m e | n u) = <m n | e u> = <e u | m n>
            V_eu = (cvB[m] @ caB_2d.T).reshape(nv, nc, na).transpose(1, 0, 2)
            # V_ue[n, e, u] = (m u | n e) = <m n | u e> = <n m | e u>
            V_ue = (caB[m] @ cvB_2d.T).reshape(na, nc, nv).transpose(1, 2, 0)
            d = (
                eps_c[m]
                + eps_c[:, None, None]
                - eps_v[None, :, None]
                - eps_a[None, None, :]
            )
            Vtilde_m = np.ascontiguousarray(V_eu * (1.0 + np.exp(-s * d * d)))
            S2_m = np.ascontiguousarray((2.0 * V_eu - V_ue) * _regularized_denom(d, s))
            # C1_ccav[w, z] += sum_{n, e} Vtilde_m[n, e, w] * S2_m[n, e, z]
            C1_ccav += Vtilde_m.reshape(nc * nv, na).T @ S2_m.reshape(nc * nv, na)

        # Energy: E += sum_{wz in sl_pa} C1_ccav[w, z] * e1_pa[z, w]
        E_ccav = float(
            np.einsum("wz,zw->", C1_ccav[sl_pa, sl_pa], e1_pa, optimize=True)
        )
        return E_ccav, C1_ccav

    @staticmethod
    def _compute_vvaa_exact(B_va, eps_a, eps_v, s, g1_ha, l2_a, sl_ha, sl_aa, blk_e=64):
        """
        Streamed, exact 2D-DGEMM evaluation of all vvaa / aavv contributions to
        delta E^{AB}, C1, and C2 without materializing full (nv, nv, na, na) tensors.

        Exploits particle-exchange symmetry V[f,e,u,v] = V[e,f,v,u] and
        T2[u,v,f,e] = T2[v,u,e,f] to loop only over the upper triangle eb <= fb
        (exact 2x FLOP reduction) and derives E_vvaa and C1_vvaa directly from
        the accumulated C2_vvaa[u, v, x, y] = sum_{e,f} T2[u,v,e,f] * Vtilde[e,f,x,y].
        """
        na = eps_a.shape[0]
        nv = eps_v.shape[0]
        C2_vvaa = np.zeros((na, na, na, na))
        if na == 0 or nv == 0:
            return 0.0, np.zeros((na, na)), C2_vvaa

        naux = B_va.shape[0]
        vaB = np.ascontiguousarray(B_va.transpose(1, 2, 0))  # (nv, na, P)

        for e0 in range(0, nv, blk_e):
            e1 = min(e0 + blk_e, nv)
            neb = e1 - e0
            eb = slice(e0, e1)
            B_eb = vaB[eb].reshape(neb * na, naux)
            for f0 in range(e0, nv, blk_e):
                f1 = min(f0 + blk_e, nv)
                nfb = f1 - f0
                fb = slice(f0, f1)
                B_fb = vaB[fb].reshape(nfb * na, naux)

                # V_blk[e, f, x, y] = sum_P B_va[P, e, x] * B_va[P, f, y]
                V_blk = (B_eb @ B_fb.T).reshape(neb, na, nfb, na).transpose(0, 2, 1, 3)

                # T2_blk[u, v, e, f]
                d_t2 = (
                    eps_a[:, None, None, None]
                    + eps_a[None, :, None, None]
                    - eps_v[None, None, eb, None]
                    - eps_v[None, None, None, fb]
                )
                T2_blk = np.ascontiguousarray(
                    V_blk.transpose(2, 3, 0, 1) * _regularized_denom(d_t2, s)
                )

                # Vtilde_blk[e, f, x, y] = V_blk * (1 + exp(-s * d^2))
                d_ef_uv = d_t2.transpose(2, 3, 0, 1)
                Vtilde_blk = np.ascontiguousarray(
                    V_blk * (1.0 + np.exp(-s * d_ef_uv * d_ef_uv))
                )

                M = (
                    T2_blk.reshape(na * na, neb * nfb)
                    @ Vtilde_blk.reshape(neb * nfb, na * na)
                ).reshape(na, na, na, na)

                C2_vvaa += M
                if f0 > e0:
                    # Pair (f, e) contributes M[v, u, y, x] by exact particle-exchange symmetry
                    C2_vvaa += M.transpose(1, 0, 3, 2)

        # Since T2[w, v, f, e] == T2[v, w, e, f]:
        # sum_{ef} S2[w, v, e, f] * Vtilde[e, f, z, u] = 2 * C2_vvaa[w, v, z, u] - C2_vvaa[v, w, z, u]
        C2_ha_full = 2.0 * C2_vvaa[:, sl_ha, :, sl_ha] - C2_vvaa[
            sl_ha, :, :, sl_ha
        ].swapaxes(0, 1)
        C1_vvaa = 0.5 * np.einsum("wvzu,uv->wz", C2_ha_full, g1_ha, optimize=True)

        # E_vvaa = 0.25 * sum_{yvxu in sl_ha} S2_C2[y,v,x,u] * g1[u,v] * g1[x,y]
        #        + 0.50 * sum_{uvxy in sl_aa} C2_vvaa[u,v,x,y] * l2_a[u,v,x,y]
        E_vvaa = 0.25 * float(
            np.einsum(
                "yvxu,uv,xy->",
                C2_ha_full[sl_ha, :, sl_ha, :],
                g1_ha,
                g1_ha,
                optimize=True,
            )
        )
        if l2_a.size > 0:
            E_vvaa += 0.5 * float(
                np.einsum(
                    "uvxy,uvxy->",
                    C2_vvaa[sl_aa, sl_aa, sl_aa, sl_aa],
                    l2_a,
                    optimize=True,
                )
            )
        return E_vvaa, C1_vvaa, C2_vvaa

    # ------------------------------------------------------------------
    # Non-streamed contractions (block-sparse in g1_ha, e1_pa, lambda2, lambda3!)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_E_AB_nonstreamed(
        h1,
        h2,
        t1,
        t2,
        s2,
        g1_ha,
        e1_pa,
        l2_a,
        l3_a,
        hc,
        ha,
        pa,
        pv,
        sl_ha,
        sl_pa,
        sl_aa,
    ):
        E = 0.0
        # H1_T1_C0 (excluding pure-environment 2 * h1[pv, hc] * t1[hc, pv] which is in delta_E_B)
        E += 2.0 * np.einsum("um,mu->", h1[pa, hc], t1[hc, pa], optimize=True)
        E += np.einsum(
            "ev,ue,uv->",
            h1[pv, ha][:, sl_ha],
            t1[ha, pv][sl_ha, :],
            g1_ha,
            optimize=True,
        )
        E -= np.einsum(
            "um,mv,uv->",
            h1[pa, hc][sl_ha, :],
            t1[hc, pa][:, sl_ha],
            g1_ha,
            optimize=True,
        )

        # H2_T2_C0 (1-RDM / 1-hole density terms; vvaa is accumulated in _compute_vvaa_exact)
        E += 0.25 * np.einsum(
            "vymn,mnux,uv,xy->",
            h2["aacc"][sl_pa, sl_pa, :, :],
            s2["ccaa"][:, :, sl_pa, sl_pa],
            e1_pa,
            e1_pa,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vemx,myue,uv,xy->",
            h2["avca"][sl_pa, :, :, sl_ha],
            s2["caav"][:, sl_ha, sl_pa, :],
            e1_pa,
            g1_ha,
            optimize=True,
        )
        E += 0.5 * np.einsum(
            "vexm,ymue,uv,xy->",
            h2["avac"][sl_pa, :, sl_ha, :],
            s2["acav"][sl_ha, :, sl_pa, :],
            e1_pa,
            g1_ha,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "evwx,zyeu,wz,uv,xy->",
            h2["vaaa"][:, sl_pa, sl_ha, sl_ha],
            s2["aava"][sl_ha, sl_ha, :, sl_pa],
            g1_ha,
            e1_pa,
            g1_ha,
            optimize=True,
        )
        E += 0.25 * np.einsum(
            "vzmx,myuw,wz,uv,xy->",
            h2["aaca"][sl_pa, sl_pa, :, sl_ha],
            s2["caaa"][:, sl_ha, sl_pa, sl_pa],
            e1_pa,
            e1_pa,
            g1_ha,
            optimize=True,
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
                h2["aaca"][sl_aa, sl_aa, :, sl_ha],
                t2["caaa"][:, sl_ha, sl_aa, sl_aa],
                g1_ha,
                l2_a,
                optimize=True,
            )
            # vvaa * l2_a term is included in _compute_vvaa_exact
            E += 0.5 * np.einsum(
                "ezxy,uvew,wz,uvxy->",
                h2["vaaa"][:, sl_pa, sl_aa, sl_aa],
                t2["aava"][sl_aa, sl_aa, :, sl_pa],
                e1_pa,
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
                h2["vaaa"][:, sl_aa, sl_ha, sl_aa],
                s2["aava"][sl_ha, sl_aa, :, sl_aa],
                g1_ha,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "euxw,zvey,wz,uvxy->",
                h2["vaaa"][:, sl_aa, sl_aa, sl_ha],
                t2["aava"][sl_ha, sl_aa, :, sl_aa],
                g1_ha,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "evxw,uzey,wz,uvxy->",
                h2["vaaa"][:, sl_aa, sl_aa, sl_ha],
                t2["aava"][sl_aa, sl_ha, :, sl_aa],
                g1_ha,
                l2_a,
                optimize=True,
            )
            E += 0.5 * np.einsum(
                "wumx,mvzy,wz,uvxy->",
                h2["aaca"][sl_pa, sl_aa, :, sl_aa],
                s2["caaa"][:, sl_aa, sl_pa, sl_aa],
                e1_pa,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "uwmx,mvzy,wz,uvxy->",
                h2["aaca"][sl_aa, sl_pa, :, sl_aa],
                t2["caaa"][:, sl_aa, sl_pa, sl_aa],
                e1_pa,
                l2_a,
                optimize=True,
            )
            E -= 0.5 * np.einsum(
                "vwmx,muyz,wz,uvxy->",
                h2["aaca"][sl_aa, sl_pa, :, sl_aa],
                t2["caaa"][:, sl_aa, sl_aa, sl_pa],
                e1_pa,
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
        h1, h2, t1, t2, s2, g1_ha, e1_pa, l2_a, hc, ha, pa, pv, sl_ha, sl_pa, sl_aa, na
    ):
        C1 = np.zeros((na, na))

        # Singles & F_tilde contractions
        C1 += np.einsum("ev,ue->uv", h1[pv, ha], t1[ha, pv], optimize=True)
        C1 -= np.einsum("um,mv->uv", h1[pa, hc], t1[hc, pa], optimize=True)
        C1 += np.einsum("em,umve->uv", h1[pv, hc], s2["acav"], optimize=True)
        C1 += np.einsum("xm,muxv->uv", h1[pa, hc], s2["caaa"], optimize=True)
        C1 += 0.50 * np.einsum(
            "ex,yuev,xy->uv",
            h1[pv, ha][:, sl_ha],
            s2["aava"][sl_ha],
            g1_ha,
            optimize=True,
        )
        C1 -= 0.50 * np.einsum(
            "ym,muxv,xy->uv",
            h1[pa, hc][sl_ha, :],
            s2["caaa"][:, :, sl_ha, :],
            g1_ha,
            optimize=True,
        )

        # Pre-transpose heavy nv-dependent 4-index tensors once for 2D BLAS DGEMM
        na = h2["aaaa"].shape[0]
        nv = h2["vaaa"].shape[0]
        nc = h2["aacc"].shape[2]
        nh_a = g1_ha.shape[0]
        np_a = e1_pa.shape[0]

        H_avca = np.ascontiguousarray(h2["avca"])  # (na, nv, nc, na)
        H_avac_uemz = np.ascontiguousarray(
            h2["avac"].transpose(0, 1, 3, 2)
        )  # (na, nv, nc, na)
        H_vaaa = np.ascontiguousarray(h2["vaaa"])  # (nv, na, na, na)
        H_vaaa_w_evu = np.ascontiguousarray(
            H_vaaa.transpose(1, 0, 2, 3)
        )  # (na, nv, na, na)

        S_caav_w_uem = np.ascontiguousarray(
            s2["caav"].transpose(1, 2, 3, 0)
        )  # (na, na, nv, nc)
        S_caav_emuz = np.ascontiguousarray(
            s2["caav"].transpose(3, 0, 1, 2)
        )  # (nv, nc, na, na)
        S_acav_w_uem = np.ascontiguousarray(
            s2["acav"].transpose(0, 2, 3, 1)
        )  # (na, na, nv, nc)
        S_acav_emuz = np.ascontiguousarray(
            s2["acav"].transpose(3, 1, 0, 2)
        )  # (nv, nc, na, na)
        S_aava_evu_z = np.ascontiguousarray(
            s2["aava"].transpose(2, 0, 1, 3)
        )  # (nv, na, na, na)

        # Pure two-body contractions (no density matrix)
        if nv > 0 and nc > 0:
            C1 += S_caav_w_uem.reshape(na, na * nv * nc) @ H_avca.reshape(
                na * nv * nc, na
            )
            C1 += S_acav_w_uem.reshape(na, na * nv * nc) @ H_avac_uemz.reshape(
                na * nv * nc, na
            )
            C1 -= H_avca.reshape(na, nv * nc * na) @ S_caav_emuz.reshape(
                nv * nc * na, na
            )
            C1 -= H_avac_uemz.reshape(na, nv * nc * na) @ S_acav_emuz.reshape(
                nv * nc * na, na
            )
        C1 += np.einsum("vumz,mwvu->wz", h2["aaca"], s2["caaa"], optimize=True)
        if nv > 0:
            C1 -= H_vaaa_w_evu.reshape(na, nv * na * na) @ S_aava_evu_z.reshape(
                nv * na * na, na
            )

        # Contractions with g1_ha (vvaa is included in C1_vvaa)
        if nh_a > 0 and nv > 0:
            S_w_exv = np.ascontiguousarray(s2["aava"][:, sl_ha].transpose(0, 2, 3, 1))
            H_exv_z = np.ascontiguousarray(
                (H_vaaa[:, :, :, sl_ha] @ g1_ha).transpose(0, 1, 3, 2)
            )
            C1 += (
                0.5
                * S_w_exv.reshape(na, nv * na * nh_a)
                @ H_exv_z.reshape(nv * na * nh_a, na)
            )

            S_w_exv2 = np.ascontiguousarray(s2["aava"][sl_ha, :].transpose(1, 2, 3, 0))
            H_exv_z2 = np.ascontiguousarray(
                (H_vaaa[:, :, sl_ha, :].transpose(0, 1, 3, 2) @ g1_ha).transpose(
                    0, 1, 3, 2
                )
            )
            C1 += (
                0.5
                * S_w_exv2.reshape(na, nv * na * nh_a)
                @ H_exv_z2.reshape(nv * na * nh_a, na)
            )

            S_tmp1 = np.ascontiguousarray(
                s2["aava"][sl_ha, :, :, sl_ha].transpose(1, 2, 0, 3)
            )
            M1_wbvx = np.ascontiguousarray(
                (g1_ha[None, None, :, :] @ S_tmp1 @ g1_ha[None, None, :, :]).transpose(
                    0, 1, 3, 2
                )
            )
            C1 -= 0.25 * (
                M1_wbvx.reshape(na, nv * nh_a * nh_a)
                @ np.ascontiguousarray(H_vaaa[:, sl_ha, sl_ha, :]).reshape(
                    nv * nh_a * nh_a, na
                )
            )

            S_tmp2 = np.ascontiguousarray(
                s2["aava"][:, sl_ha, :, sl_ha].transpose(0, 2, 1, 3)
            )
            M2_wbvx = np.ascontiguousarray(
                (g1_ha[None, None, :, :] @ S_tmp2 @ g1_ha[None, None, :, :]).transpose(
                    0, 1, 3, 2
                )
            )
            H_bvx_z = np.ascontiguousarray(
                H_vaaa[:, sl_ha, :, sl_ha].transpose(0, 1, 3, 2)
            )
            C1 -= 0.25 * (
                M2_wbvx.reshape(na, nv * nh_a * nh_a)
                @ H_bvx_z.reshape(nv * nh_a * nh_a, na)
            )

        if nh_a > 0 and nv > 0 and nc > 0:
            H_avac_g = (g1_ha @ H_avac_uemz[sl_ha].reshape(nh_a, -1)).reshape(
                nh_a * nv * nc, na
            )
            C1 -= (
                0.5
                * np.ascontiguousarray(S_acav_w_uem[:, sl_ha]).reshape(
                    na, nh_a * nv * nc
                )
                @ H_avac_g
            )

            H_avca_g = (g1_ha @ H_avca[sl_ha].reshape(nh_a, -1)).reshape(
                nh_a * nv * nc, na
            )
            C1 -= (
                0.5
                * np.ascontiguousarray(S_caav_w_uem[:, sl_ha]).reshape(
                    na, nh_a * nv * nc
                )
                @ H_avca_g
            )

        C1 -= 0.5 * np.einsum(
            "mwxu,xvmz,uv->wz",
            s2["caaa"][:, :, :, sl_ha],
            h2["aaca"][:, sl_ha],
            g1_ha,
            optimize=True,
        )
        C1 -= 0.5 * np.einsum(
            "mwux,vxmz,uv->wz",
            s2["caaa"][:, :, sl_ha, :],
            h2["aaca"][sl_ha],
            g1_ha,
            optimize=True,
        )
        C1 += 0.25 * np.einsum(
            "jwxu,xy,yvjz,uv->wz",
            s2["caaa"][:, :, sl_ha, sl_ha],
            g1_ha,
            h2["aaca"][sl_ha, sl_ha],
            g1_ha,
            optimize=True,
        )

        # Contractions with e1_pa
        C1 -= 0.5 * np.einsum(
            "mnzu,wvmn,uv->wz",
            s2["ccaa"][:, :, :, sl_pa],
            h2["aacc"][:, sl_pa],
            e1_pa,
            optimize=True,
        )
        C1 -= 0.5 * np.einsum(
            "mxzu,wvmx,uv->wz",
            s2["caaa"][:, :, :, sl_pa],
            h2["aaca"][:, sl_pa],
            e1_pa,
            optimize=True,
        )
        C1 -= 0.5 * np.einsum(
            "mxuz,vwmx,uv->wz",
            s2["caaa"][:, :, sl_pa, :],
            h2["aaca"][sl_pa],
            e1_pa,
            optimize=True,
        )
        if np_a > 0 and nv > 0 and nc > 0:
            H_avac_e = (H_avac_uemz[:, :, :, sl_pa] @ e1_pa).reshape(na, nv * nc * np_a)
            C1 += (
                0.5
                * H_avac_e
                @ np.ascontiguousarray(S_acav_emuz[:, :, sl_pa, :]).reshape(
                    nv * nc * np_a, na
                )
            )

            H_avca_e = (H_avca[:, :, :, sl_pa] @ e1_pa).reshape(na, nv * nc * np_a)
            C1 += (
                0.5
                * H_avca_e
                @ np.ascontiguousarray(S_caav_emuz[:, :, sl_pa, :]).reshape(
                    nv * nc * np_a, na
                )
            )

        if np_a > 0 and nv > 0:
            H_vaaa_e1 = (H_vaaa_w_evu[:, :, :, sl_pa] @ e1_pa).reshape(
                na, nv * na * np_a
            )
            C1 += (
                0.5
                * H_vaaa_e1
                @ np.ascontiguousarray(S_aava_evu_z[:, :, sl_pa, :]).reshape(
                    nv * na * np_a, na
                )
            )

            H_vaaa_e2 = np.ascontiguousarray(
                e1_pa.T[None, None, :, :] @ H_vaaa_w_evu[:, :, sl_pa, :]
            ).reshape(na, nv * np_a * na)
            C1 += (
                0.5
                * H_vaaa_e2
                @ np.ascontiguousarray(S_aava_evu_z[:, sl_pa, :, :]).reshape(
                    nv * np_a * na, na
                )
            )

            H_wbyv = np.ascontiguousarray(
                e1_pa.T[None, None, :, :]
                @ H_vaaa_w_evu[:, :, sl_pa, sl_pa]
                @ e1_pa[None, None, :, :]
            ).reshape(na, nv * np_a * np_a)
            C1 -= (
                0.25
                * H_wbyv
                @ np.ascontiguousarray(S_aava_evu_z[:, sl_pa, sl_pa, :]).reshape(
                    nv * np_a * np_a, na
                )
            )

        C1 += 0.25 * np.einsum(
            "jvxz,xy,ywju,uv->wz",
            s2["caaa"][:, sl_pa, sl_pa, :],
            e1_pa,
            h2["aaca"][sl_pa, :, :, sl_pa],
            e1_pa,
            optimize=True,
        )
        C1 += 0.25 * np.einsum(
            "jvzx,xy,wyju,uv->wz",
            s2["caaa"][:, sl_pa, :, sl_pa],
            e1_pa,
            h2["aaca"][:, sl_pa, :, sl_pa],
            e1_pa,
            optimize=True,
        )

        # G2 intermediates for T1 contractions (avoid materializing 4D G2_avac / G2_avaa)
        G2_aaac = 2.0 * h2["aaca"].transpose(1, 0, 3, 2) - h2["aaca"].transpose(
            0, 1, 3, 2
        )
        C1 += np.einsum("ma,uavm->uv", t1[hc, pa], G2_aaac, optimize=True)
        if nv > 0 and nc > 0:
            t1_em = np.ascontiguousarray(t1[hc, pv].T)
            C1 += 2.0 * np.tensordot(H_avac_uemz, t1_em, axes=([1, 2], [0, 1]))
            C1 -= np.tensordot(H_avca, t1_em, axes=([1, 2], [0, 1]))
        if nh_a > 0 and nv > 0:
            M_ye = g1_ha @ t1[ha, pv][sl_ha, :]
            C1 += np.tensordot(M_ye, H_vaaa[:, :, sl_ha, :], axes=([0, 1], [2, 0]))
            C1 -= 0.50 * np.tensordot(
                M_ye, H_vaaa[:, :, :, sl_ha], axes=([0, 1], [3, 0])
            )
        C1 -= 0.50 * np.einsum(
            "mx,xy,uyvm->uv",
            t1[hc, pa][:, sl_ha],
            g1_ha,
            G2_aaac[:, sl_ha, :, :],
            optimize=True,
        )

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
            if nv > 0:
                M_xe = np.einsum(
                    "uvey,xyuv->xe",
                    t2["aava"][sl_aa, sl_aa, :, sl_aa],
                    l2_a,
                    optimize=True,
                )
                C1 += np.tensordot(M_xe, H_vaaa[:, :, sl_aa, :], axes=([0, 1], [2, 0]))
                C1 -= 0.50 * np.tensordot(
                    M_xe, H_vaaa[:, :, :, sl_aa], axes=([0, 1], [3, 0])
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
    def _compute_C2(h1, h2, t1, t2, s2, g1_ha, sl_ha, hc, ha, pa, pv, na, C2_vvaa):
        C2 = C2_vvaa.copy()
        nc = hc.stop - hc.start
        nv = pv.stop - pv.start
        nh_a = sl_ha.stop - sl_ha.start

        if nv > 0:
            M_va = (
                t2["aava"].reshape(na * na, nv * na)
                @ h2["vaaa"].reshape(nv * na, na * na)
            ).reshape(na, na, na, na)
            C2 += M_va
            C2 += M_va.transpose(1, 0, 3, 2)

        if nc > 0:
            C2 += (
                h2["aacc"].reshape(na * na, nc * nc)
                @ t2["ccaa"].reshape(nc * nc, na * na)
            ).reshape(na, na, na, na)
            M_ca = (
                h2["aaca"].reshape(na * na, nc * na)
                @ t2["caaa"].reshape(nc * na, na * na)
            ).reshape(na, na, na, na)
            C2 += M_ca
            C2 += M_ca.transpose(1, 0, 3, 2)
        else:
            M_ca = None

        temp = np.einsum("ax,uvay->uvxy", h1[pv, ha], t2["aava"], optimize=True)
        temp -= np.einsum("ui,ivxy->uvxy", h1[pa, hc], t2["caaa"], optimize=True)
        temp += np.einsum("ua,avxy->uvxy", t1[ha, pv], h2["vaaa"], optimize=True)
        temp -= np.einsum("ix,uviy->uvxy", t1[hc, pa], h2["aaca"], optimize=True)

        # -0.5 * sum_{w,z in sl_ha} g1[w,z] * t2["aava"][v,u,a,w] * h2["vaaa"][a,z,y,x]
        if nv > 0 and nh_a > 0:
            tg = np.ascontiguousarray(
                np.einsum(
                    "wz,vuaw->uvaz", g1_ha, t2["aava"][:, :, :, sl_ha], optimize=True
                )
            )
            h2_az = np.ascontiguousarray(
                h2["vaaa"][:, sl_ha, :, :].transpose(0, 1, 3, 2)
            )
            temp -= 0.50 * (
                tg.reshape(na * na, nv * nh_a) @ h2_az.reshape(nv * nh_a, na * na)
            ).reshape(na, na, na, na)

        # -0.5 * sum_{w,z} e1[w,z] * t2["caaa"][i,z,y,x] * h2["aaca"][v,u,i,w]
        # Substitute e1 = 2*I - g1:
        #   - sum_{i,w} h2[v,u,i,w] * t2[i,w,y,x]  (= -M_ca.transpose(1,0,3,2))
        #   + 0.5 * sum_{w,z in sl_ha} g1[w,z] * h2[v,u,i,w] * t2[i,z,y,x]
        if nc > 0:
            temp -= M_ca.transpose(1, 0, 3, 2)
            if nh_a > 0:
                hg = np.ascontiguousarray(
                    np.einsum(
                        "wz,vuiw->uviz",
                        g1_ha,
                        h2["aaca"][:, :, :, sl_ha],
                        optimize=True,
                    )
                )
                t_iz = np.ascontiguousarray(
                    t2["caaa"][:, sl_ha, :, :].transpose(0, 1, 3, 2)
                )
                temp += 0.50 * (
                    hg.reshape(na * na, nc * nh_a) @ t_iz.reshape(nc * nh_a, na * na)
                ).reshape(na, na, na, na)

        if nc > 0 and nv > 0:
            temp += np.einsum("uexm,vmye->uvxy", h2["avac"], s2["acav"], optimize=True)
            temp -= np.einsum("uemx,vmye->uvxy", h2["avca"], t2["acav"], optimize=True)
            temp -= np.einsum("vemx,muye->uvxy", h2["avca"], t2["caav"], optimize=True)

        if nc > 0:
            # + einsum("wumx,mvwy->uvxy", h2["aaca"], s2["caaa"])
            H_wmx = np.ascontiguousarray(h2["aaca"].transpose(1, 3, 0, 2)).reshape(
                na * na, na * nc
            )
            S_mw = np.ascontiguousarray(s2["caaa"].transpose(2, 0, 1, 3)).reshape(
                na * nc, na * na
            )
            temp += (H_wmx @ S_mw).reshape(na, na, na, na).transpose(0, 2, 1, 3)

            # - einsum("uwmx,mvwy->uvxy", h2["aaca"], t2["caaa"])
            # - einsum("vwmx,muyw->uvxy", h2["aaca"], t2["caaa"])
            H_uw = np.ascontiguousarray(h2["aaca"].transpose(0, 3, 1, 2)).reshape(
                na * na, na * nc
            )
            T_mw = np.ascontiguousarray(t2["caaa"].transpose(2, 0, 1, 3)).reshape(
                na * nc, na * na
            )
            temp -= (H_uw @ T_mw).reshape(na, na, na, na).transpose(0, 2, 1, 3)
            T_mw2 = np.ascontiguousarray(t2["caaa"].transpose(3, 0, 1, 2)).reshape(
                na * nc, na * na
            )
            temp -= (H_uw @ T_mw2).reshape(na, na, na, na).transpose(2, 0, 1, 3)

            if nh_a > 0:
                # - 0.50 * einsum("wz,ivwy,zuix->uvxy", g1_ha, s2["caaa"][:, :, sl_ha], h2["aaca"][sl_ha])
                sg_c = np.ascontiguousarray(
                    np.einsum(
                        "wz,ivwy->izvy",
                        g1_ha,
                        s2["caaa"][:, :, sl_ha, :],
                        optimize=True,
                    )
                ).reshape(nc * nh_a, na * na)
                H_zi = np.ascontiguousarray(
                    h2["aaca"][sl_ha].transpose(1, 3, 2, 0)
                ).reshape(na * na, nc * nh_a)
                temp -= 0.50 * (H_zi @ sg_c).reshape(na, na, na, na).transpose(
                    0, 2, 1, 3
                )

                # + 0.50 * einsum("wz,ivwy,uzix->uvxy", g1_ha, t2["caaa"][:, :, sl_ha], h2["aaca"][:, sl_ha])
                # + 0.50 * einsum("wz,iuyw,vzix->uvxy", g1_ha, t2["caaa"][:, :, :, sl_ha], h2["aaca"][:, sl_ha])
                H_zi2 = np.ascontiguousarray(
                    h2["aaca"][:, sl_ha].transpose(0, 3, 2, 1)
                ).reshape(na * na, nc * nh_a)
                tg_c1 = np.ascontiguousarray(
                    np.einsum(
                        "wz,ivwy->izvy",
                        g1_ha,
                        t2["caaa"][:, :, sl_ha, :],
                        optimize=True,
                    )
                ).reshape(nc * nh_a, na * na)
                temp += 0.50 * (H_zi2 @ tg_c1).reshape(na, na, na, na).transpose(
                    0, 2, 1, 3
                )
                tg_c2 = np.ascontiguousarray(
                    np.einsum(
                        "wz,iuyw->izuy",
                        g1_ha,
                        t2["caaa"][:, :, :, sl_ha],
                        optimize=True,
                    )
                ).reshape(nc * nh_a, na * na)
                temp += 0.50 * (H_zi2 @ tg_c2).reshape(na, na, na, na).transpose(
                    2, 0, 1, 3
                )

        if nv > 0 and nh_a > 0:
            # + 0.50 * einsum("wz,zvay,auwx->uvxy", g1_ha, s2["aava"][sl_ha], h2["vaaa"][:, :, sl_ha, :])
            H_aw1 = np.ascontiguousarray(
                h2["vaaa"][:, :, sl_ha, :].transpose(1, 3, 0, 2)
            ).reshape(na * na, nv * nh_a)
            S_aw1 = np.ascontiguousarray(
                np.einsum("wz,zvay->awvy", g1_ha, s2["aava"][sl_ha], optimize=True)
            ).reshape(nv * nh_a, na * na)
            temp += 0.50 * (H_aw1 @ S_aw1).reshape(na, na, na, na).transpose(0, 2, 1, 3)

            # - 0.50 * einsum("wz,zvay,auxw->uvxy", g1_ha, t2["aava"][sl_ha], h2["vaaa"][:, :, :, sl_ha])
            # - 0.50 * einsum("wz,uzay,avxw->uvxy", g1_ha, t2["aava"][:, sl_ha], h2["vaaa"][:, :, :, sl_ha])
            H_aw2 = np.ascontiguousarray(
                h2["vaaa"][:, :, :, sl_ha].transpose(1, 2, 0, 3)
            ).reshape(na * na, nv * nh_a)
            T_aw1 = np.ascontiguousarray(
                np.einsum("wz,zvay->awvy", g1_ha, t2["aava"][sl_ha], optimize=True)
            ).reshape(nv * nh_a, na * na)
            temp -= 0.50 * (H_aw2 @ T_aw1).reshape(na, na, na, na).transpose(0, 2, 1, 3)
            T_aw2 = np.ascontiguousarray(
                np.einsum("wz,uzay->awuy", g1_ha, t2["aava"][:, sl_ha], optimize=True)
            ).reshape(nv * nh_a, na * na)
            temp -= 0.50 * (H_aw2 @ T_aw2).reshape(na, na, na, na).transpose(2, 0, 1, 3)

        C2 += temp
        C2 += temp.transpose(1, 0, 3, 2)
        return C2
