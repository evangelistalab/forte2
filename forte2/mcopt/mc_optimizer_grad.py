import numpy as np
from numpy.typing import NDArray

from forte2._forte2 import ints
from forte2.base_classes import RelCIBase
from forte2.gradients import (
    build_metric_inverted_three_center,
    compute_gradient,
)
from forte2.system import ModelSystem

from .orbital_optimizer import OrbOptimizer


def _compute_casscf_gradient(mc) -> NDArray:
    """Compute a state-specific density-fitted CASSCF/GASSCF gradient."""
    _validate_casscf_gradient_supported(mc, pre_run=True)

    if not mc.executed:
        mc.run()

    _validate_casscf_gradient_supported(mc, pre_run=False)

    C = mc.C[0][:, mc.mo_space.orig_to_contig].copy()
    gamma1_act = mc.make_sf_1rdm(0)
    gamma2_act = mc.make_sf_2rdm(0)
    Ccore = C[:, mc.mo_space.core]
    Cact = C[:, mc.mo_space.actv]

    D1 = _build_casscf_one_body_density(Ccore, Cact, gamma1_act)
    W1 = _build_casscf_overlap_weight(mc, C, gamma1_act, gamma2_act)
    W2, W3 = _build_casscf_df_deriv_weights(
        mc.system, Ccore, Cact, gamma1_act, gamma2_act
    )

    return compute_gradient(mc.system, D1.real, W1.real, W2, W3)


def _build_casscf_one_body_density(
    Ccore: NDArray,
    Cact: NDArray,
    gamma1_act: NDArray,
) -> NDArray:
    r"""
    Build the AO spin-free one-particle density without full MO padding.

    For the first CASSCF/GASSCF gradient implementation, frozen core
    orbitals are rejected and the inactive core is a closed-shell doubly
    occupied block.  The spin-free AO density is therefore assembled
    directly as

    .. math::
        \Gamma_{\mu\nu}
        =
        2 C_{\mu i} C_{\nu i}
        +
        C_{\mu u}\Gamma_{uv} C_{\nu v},

    where :math:`i` labels inactive core orbitals and :math:`u,v` label
    active orbitals.  This avoids constructing a full
    :math:`\Gamma_{pq}` matrix over core, active, and virtual orbitals.

    Parameters
    ----------
    Ccore : NDArray
        Core MO coefficients with shape ``(nbasis, ncore)``.
    Cact : NDArray
        Active MO coefficients with shape ``(nbasis, nactv)``.
    gamma1_act : NDArray
        Active-space spin-free 1-RDM, :math:`\Gamma_{uv}`.

    Returns
    -------
    NDArray
        AO spin-free one-particle density with shape ``(nbasis, nbasis)``.
    """
    nact = Cact.shape[1]
    if gamma1_act.shape != (nact, nact):
        raise ValueError(
            f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
        )

    D1 = 2.0 * np.einsum("mi,ni->mn", Ccore, Ccore.conj(), optimize=True)
    D1 += np.einsum("mu,uv,nv->mn", Cact, gamma1_act, Cact.conj(), optimize=True)
    return D1


def _build_casscf_active_cumulant(
    gamma1_act: NDArray,
    gamma2_act: NDArray,
) -> NDArray:
    r"""
    Build the active-space spin-free two-particle cumulant.

    The CASSCF/GASSCF pair density is written in the notation of
    ``docs/technical_notes/df_gradients.tex`` as

    .. math::
        \Gamma_{pq,rs}
        =
        \Gamma_{pr}\Gamma_{qs}
        -
        \frac{1}{2}\Gamma_{ps}\Gamma_{qr}
        +
        \Lambda_{pq,rs}.

    The inactive core contribution is generated from the one-particle
    density and is never materialized as a full four-index tensor.  The
    genuine correlated contribution is the active cumulant

    .. math::
        \Lambda_{uv,wx}
        =
        \Gamma_{uv,wx}
        -
        \Gamma_{uw}\Gamma_{vx}
        +
        \frac{1}{2}\Gamma_{ux}\Gamma_{vw}.

    Parameters
    ----------
    gamma1_act : NDArray
        Active-space spin-free 1-RDM.
    gamma2_act : NDArray
        Active-space spin-free 2-RDM in the Forte2 CI convention.

    Returns
    -------
    NDArray
        Active-space spin-free two-particle cumulant.
    """
    gamma1_act = np.asarray(gamma1_act)
    gamma2_act = np.asarray(gamma2_act)
    nact = gamma1_act.shape[0]
    if gamma1_act.shape != (nact, nact):
        raise ValueError("Active 1-RDM must be a square matrix.")
    if gamma2_act.shape != (nact, nact, nact, nact):
        raise ValueError(
            "Expected active 2-RDM shape "
            f"{(nact, nact, nact, nact)}, got {gamma2_act.shape}."
        )

    lambda2_act = gamma2_act.copy()
    lambda2_act -= np.einsum("uw,vx->uvwx", gamma1_act, gamma1_act, optimize=True)
    lambda2_act += 0.5 * np.einsum("ux,vw->uvwx", gamma1_act, gamma1_act, optimize=True)
    return lambda2_act


def _build_casscf_df_deriv_weights(
    system,
    Ccore: NDArray,
    Cact: NDArray,
    gamma1_act: NDArray,
    gamma2_act: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""
    Build CASSCF/GASSCF DF derivative weights from core and active blocks only.

    This implements the molecular-orbital DF derivative equations from
    ``docs/technical_notes/df_gradients.tex`` without constructing a full
    orbital-space :math:`\Gamma_{pq,rs}` tensor.  Let
    :math:`x,y,z,w` run only over the compact hole space containing the
    inactive core and active orbitals,

    .. math::
        C_h = [C_i\ C_u],
        \qquad
        \Gamma^h =
        \begin{pmatrix}
        2I_\mathrm{core} & 0 \\
        0 & \Gamma^\mathrm{act}
        \end{pmatrix}.

    The metric-applied three-center tensor is transformed only to this
    compact space:

    .. math::
        Z^P_{xy} = C_{\mu x} Z^P_{\mu\nu} C_{\nu y}.

    The product part of the spin-free pair density contributes through
    :math:`\Gamma^h`, while the cumulant correction is nonzero only in the
    active block.  The resulting weights are

    .. math::
        W^P_{xy}
        =
        \Gamma^h_{xy} R^P
        -
        \frac{1}{2}
        \Gamma^h_{xz} Z^P_{wz} \Gamma^h_{wy}
        +
        \delta_{xu}\delta_{yw}
        \Lambda_{uv,wx}Z^P_{vx},

    and

    .. math::
        W_{PQ}
        =
        -\frac{1}{2}R^P R^Q
        +
        \frac{1}{4}
        Z^P_{xy}\Gamma^h_{xz}Z^Q_{wz}\Gamma^h_{wy}
        -
        \frac{1}{2}
        \Lambda_{uv,wx}Z^P_{uw}Z^Q_{vx},

    with :math:`R^P=\Gamma^h_{xy}Z^P_{xy}`.  Only the compact
    :math:`W^P_{xy}` is formed before the final AO back-transformation
    required by the derivative integral kernel.

    Parameters
    ----------
    system : System
        Molecular system providing the AO and auxiliary bases.
    Ccore : NDArray
        Core MO coefficients with shape ``(nbasis, ncore)``.
    Cact : NDArray
        Active MO coefficients with shape ``(nbasis, nactv)``.
    gamma1_act : NDArray
        Active-space spin-free 1-RDM.
    gamma2_act : NDArray
        Active-space spin-free 2-RDM.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(W2, W3)`` with shapes ``(naux, naux)`` and
        ``(naux, nbasis, nbasis)``.
    """
    ncore = Ccore.shape[1]
    nact = Cact.shape[1]
    if gamma1_act.shape != (nact, nact):
        raise ValueError(
            f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
        )

    Ch = np.hstack((Ccore, Cact))
    nhole = Ch.shape[1]
    gamma_h = np.zeros((nhole, nhole), dtype=np.result_type(gamma1_act, gamma2_act))
    gamma_h[:ncore, :ncore] = 2.0 * np.eye(ncore)
    gamma_h[ncore:, ncore:] = gamma1_act
    lambda2_act = _build_casscf_active_cumulant(gamma1_act, gamma2_act)

    Z_ao = build_metric_inverted_three_center(system)
    Z_h = np.einsum("mx,Pmn,ny->Pxy", Ch.conj(), Z_ao, Ch, optimize=True)

    R = np.einsum("xy,Pxy->P", gamma_h, Z_h, optimize=True)
    W3_h = np.einsum("xy,P->Pxy", gamma_h, R, optimize=True)
    W3_h -= 0.5 * np.einsum("xz,Pwz,wy->Pxy", gamma_h, Z_h, gamma_h, optimize=True)

    Z_act = Z_h[:, ncore:, ncore:]
    W3_h[:, ncore:, ncore:] += np.einsum(
        "uvwx,Pvx->Puw", lambda2_act, Z_act, optimize=True
    )

    W2 = -0.5 * np.einsum("P,Q->PQ", R, R, optimize=True)
    W2 += 0.25 * np.einsum(
        "Pxy,xz,Qwz,wy->PQ", Z_h, gamma_h, Z_h, gamma_h, optimize=True
    )
    W2 -= 0.5 * np.einsum("uvwx,Puw,Qvx->PQ", lambda2_act, Z_act, Z_act, optimize=True)

    W3 = np.einsum("mx,Pxy,ny->Pmn", Ch, W3_h, Ch.conj(), optimize=True)
    return W2.real, W3.real


def _build_casscf_overlap_weight(
    mc,
    C: NDArray,
    gamma1_act: NDArray,
    gamma2_act: NDArray,
) -> NDArray:
    r"""
    Build the AO energy-weighted density for the CASSCF/GASSCF overlap term.

    The existing orbital optimizer constructs the matrix :math:`A_{pq}`
    used in the CASSCF/GASSCF orbital gradient
    :math:`g_{pq}=2(A_{pq}-A_{qp})`.  For a fully optimized
    state-specific CASSCF/GASSCF wave function, the symmetric part of this
    matrix is the orbital Lagrange multiplier that contracts the overlap
    derivative.  This helper recomputes :math:`A` in the current final MO
    basis and transforms

    .. math::
        W^S_{\mu\nu}
        =
        C_{\mu p}
        \frac{1}{2}(A_{pq}+A_{qp})
        C_{\nu q}.

    Parameters
    ----------
    mc : MCOptimizer
        Converged multiconfigurational optimizer.
    C : NDArray
        Final MO coefficients in contiguous CASSCF order.
    gamma1_act : NDArray
        State-specific active-space spin-free 1-RDM.
    gamma2_act : NDArray
        State-specific active-space spin-free 2-RDM.

    Returns
    -------
    NDArray
        AO energy-weighted density with shape ``(nbasis, nbasis)``.
    """
    orb_opt = OrbOptimizer(
        C,
        (mc.mo_space.core, mc.mo_space.actv, mc.mo_space.virt),
        mc.system.fock_builder,
        mc.system.ints_hcore(),
        mc.system.nuclear_repulsion,
        mc.nrr,
        compute_active_hessian=False,
    )
    orb_opt.set_rdms(gamma1_act, gamma2_act)
    orb_opt._compute_Fcore()
    orb_opt.get_eri_gaaa()
    lagrangian_mo = orb_opt.compute_orbital_lagrangian()
    return np.einsum("mp,pq,nq->mn", C, lagrangian_mo, C, optimize=True)


def _validate_casscf_gradient_supported(mc, pre_run: bool = False) -> None:
    """Reject CASSCF/GASSCF gradient cases outside the first implementation scope."""
    if isinstance(mc.ci_solver, RelCIBase):
        raise NotImplementedError(
            "CASSCF/GASSCF gradients are implemented only for nonrelativistic "
            "CASSCF/GASSCF."
        )

    if mc.ci_solver.sa_info.ncis != 1 or mc.ci_solver.sa_info.nroots_sum != 1:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients are implemented only for state-specific "
            "CASSCF/GASSCF."
        )

    if mc.active_frozen_orbitals:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with active frozen orbitals are not implemented."
        )
    if mc.freeze_inter_gas_rots and _ci_solver_requests_multiple_gas(mc.ci_solver):
        raise NotImplementedError(
            "GASSCF gradients with frozen inter-GAS rotations are not implemented."
        )

    system = mc.system if hasattr(mc, "system") else mc.parent_method.system
    if isinstance(system, ModelSystem):
        raise NotImplementedError(
            "CASSCF/GASSCF gradients are not implemented for ModelSystem."
        )
    if system.cholesky_tei:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients are implemented only for density fitting, "
            "not cholesky_tei."
        )
    if system.use_gaussian_charges:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with Gaussian nuclear charges are not implemented."
        )
    if system.x2c_type is not None or system.two_component:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with X2C are not implemented."
        )
    if system.auxiliary_basis is None:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients require an auxiliary basis set for density "
            "fitting."
        )

    max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
    if max_l > ints.libint2_max_am:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients require derivative integrals supported by Libint2 "
            f"(max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
        )

    frozen_core = getattr(mc.ci_solver, "frozen_core_orbitals", None)
    frozen_virt = getattr(mc.ci_solver, "frozen_virtual_orbitals", None)
    if frozen_core:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with frozen core orbitals are not implemented."
        )
    if frozen_virt:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with frozen virtual orbitals are not implemented."
        )

    if pre_run:
        return

    if mc.mo_space.nfrozen_core > 0:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with frozen core orbitals are not implemented."
        )
    if mc.mo_space.nfrozen_virtual > 0:
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with frozen virtual orbitals are not implemented."
        )
    if mc.mo_space.ngas > 1 and mc.freeze_inter_gas_rots:
        raise NotImplementedError(
            "GASSCF gradients with frozen inter-GAS rotations are not implemented."
        )
    if np.iscomplexobj(mc.C[0]):
        raise NotImplementedError(
            "CASSCF/GASSCF gradients with complex orbitals are not implemented."
        )


def _ci_solver_requests_multiple_gas(ci_solver) -> bool:
    """Return whether the attached CI solver was configured with multiple GASes."""
    active_orbitals = getattr(ci_solver, "active_orbitals", None)
    if isinstance(active_orbitals, tuple):
        return len(active_orbitals) > 1
    if isinstance(active_orbitals, list):
        return (
            any(isinstance(space, list) for space in active_orbitals)
            and len(active_orbitals) > 1
        )
    mo_space = getattr(ci_solver, "mo_space", None)
    return mo_space is not None and mo_space.ngas > 1
