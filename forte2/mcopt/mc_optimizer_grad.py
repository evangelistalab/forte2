import numpy as np
from numpy.typing import NDArray

from forte2.base_classes import RelCIBase
from forte2.ci.ci_utils import make_2cumulant_so
from forte2.gradients import (
    build_metric_inverted_three_center,
    compute_gradient,
)
from forte2.gradients.validation import validate_df_gradient_system

from .orbital_optimizer import OrbOptimizer, RelOrbOptimizer


def _compute_casscf_gradient(mc) -> NDArray:
    """Compute a state-specific density-fitted CASSCF/GASSCF gradient."""
    _validate_casscf_gradient_request(mc)

    if not mc.executed:
        mc.run()

    _validate_converged_casscf_gradient(mc)

    C = mc.C[0][:, mc.mo_space.orig_to_contig].copy()
    if isinstance(mc.ci_solver, RelCIBase):
        return _compute_rel_casscf_gradient(mc, C)
    return _compute_nonrel_casscf_gradient(mc, C)


def _compute_nonrel_casscf_gradient(mc, C: NDArray) -> NDArray:
    """Compute a state-specific nonrelativistic CASSCF/GASSCF gradient."""
    gamma1_act = mc.make_sf_1rdm(0)
    gamma2_act = mc.make_sf_2rdm(0)
    Ccore = C[:, mc.mo_space.core]
    Cact = C[:, mc.mo_space.actv]

    D1 = _build_casscf_one_body_density(Ccore, Cact, gamma1_act)
    W1 = _build_casscf_overlap_weight(mc, C, gamma1_act, gamma2_act)
    W2, W3 = _build_casscf_df_deriv_weights(
        mc.system, Ccore, Cact, gamma1_act, gamma2_act
    )

    hcore_gradient = (
        mc.system.x2c_helper.hcore_gradient(D1)
        if mc.system.x2c_type is not None
        else None
    )
    return compute_gradient(
        mc.system,
        D1.real,
        W1.real,
        W2,
        W3,
        hcore_gradient=hcore_gradient,
    )


def _compute_rel_casscf_gradient(mc, C: NDArray) -> NDArray:
    """Compute a state-specific two-component CASSCF/GASSCF gradient."""
    gamma1_act = mc.ci_solver.make_1rdm(0)
    gamma2_act = mc.ci_solver.make_2rdm(0)
    Ccore = C[:, mc.mo_space.core]
    Cact = C[:, mc.mo_space.actv]

    D_spinor = _build_rel_casscf_one_body_density(Ccore, Cact, gamma1_act)
    D1 = _spatial_spin_trace(D_spinor, mc.system.nbf)
    W_spinor = _build_casscf_overlap_weight(mc, C, gamma1_act, gamma2_act)
    W1 = _spatial_spin_trace(W_spinor, mc.system.nbf)
    W2, W3 = _build_rel_casscf_df_deriv_weights(
        mc.system, Ccore, Cact, gamma1_act, gamma2_act
    )

    hcore_gradient = None
    if mc.system.x2c_type is not None:
        x2c_density = D_spinor if mc.system.x2c_type == "so" else D1
        hcore_gradient = mc.system.x2c_helper.hcore_gradient(x2c_density)

    return compute_gradient(
        mc.system,
        D1.real,
        W1.real,
        W2,
        W3,
        hcore_gradient=hcore_gradient,
    )


def _spatial_spin_trace(matrix: NDArray, nbf: int) -> NDArray:
    """Trace a spinor AO matrix over its alpha and beta diagonal blocks."""
    matrix = np.asarray(matrix)
    if matrix.shape != (2 * nbf, 2 * nbf):
        raise ValueError(
            f"Expected a spinor AO matrix with shape {(2 * nbf, 2 * nbf)}, "
            f"got {matrix.shape}."
        )
    return matrix[:nbf, :nbf] + matrix[nbf:, nbf:]


def _build_rel_casscf_one_body_density(
    Ccore: NDArray,
    Cact: NDArray,
    gamma1_act: NDArray,
) -> NDArray:
    r"""Build the full spinor AO one-particle density.

    Each inactive core spinor has unit occupation. Correlated active-space
    occupations are supplied by the spin-orbital 1-RDM. The active RDM is
    transposed because Forte2 contracts relativistic MO integrals and RDMs
    element by element.
    """
    nact = Cact.shape[1]
    if gamma1_act.shape != (nact, nact):
        raise ValueError(
            f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
        )

    D1 = np.einsum("mi,ni->mn", Ccore, Ccore.conj(), optimize=True)
    D1 += np.einsum("mu,vu,nv->mn", Cact, gamma1_act, Cact.conj(), optimize=True)
    return D1


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


def _build_mc_df_hole_weights(
    gamma_h: NDArray,
    lambda2_act: NDArray,
    Z_h: NDArray,
    ncore: int,
    exchange_factor: float,
) -> tuple[NDArray, NDArray]:
    r"""Build metric and three-center weights in the compact hole space.

    The spin-free and spin-orbital CASSCF expressions differ only in the
    exchange prefactor.  Forming the exchange and active-cumulant responses
    once also lets the metric weight reuse the intermediates required by the
    three-center weight.
    """
    R = np.einsum("xy,Pxy->P", gamma_h, Z_h, optimize=True)
    exchange_h = np.einsum("xz,Pwz,wy->Pxy", gamma_h, Z_h, gamma_h, optimize=True)

    active = slice(ncore, None)
    Z_act = Z_h[:, active, active]
    cumulant_h = np.einsum("uvwx,Pvx->Puw", lambda2_act, Z_act, optimize=True)

    W3_h = np.einsum("xy,P->Pxy", gamma_h, R, optimize=True)
    W3_h -= exchange_factor * exchange_h
    W3_h[:, active, active] += cumulant_h

    W2 = -0.5 * np.einsum("P,Q->PQ", R, R, optimize=True)
    W2 += (
        0.5 * exchange_factor * np.einsum("Pxy,Qxy->PQ", Z_h, exchange_h, optimize=True)
    )
    W2 -= 0.5 * np.einsum("Puw,Quw->PQ", Z_act, cumulant_h, optimize=True)
    return W2, W3_h


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

    W2, W3_h = _build_mc_df_hole_weights(
        gamma_h,
        lambda2_act,
        Z_h,
        ncore,
        exchange_factor=0.5,
    )

    W3 = np.einsum("mx,Pxy,ny->Pmn", Ch, W3_h, Ch.conj(), optimize=True)
    return W2.real, W3.real


def _build_rel_casscf_df_deriv_weights(
    system,
    Ccore: NDArray,
    Cact: NDArray,
    gamma1_act: NDArray,
    gamma2_act: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""Build two-component CASSCF DF derivative weights.

    The hole space contains unit-occupied inactive core spinors and the
    correlated active spinors. The spin-orbital pair density is decomposed as

    .. math::
        \Gamma_{xy,zw}
        = \gamma_{xz}\gamma_{yw}
        - \gamma_{xw}\gamma_{yz}
        + \lambda_{xy,zw}.

    Spatial three-center integrals are transformed with the sum over equal
    alpha and beta spin blocks. This retains spin mixing and complex phases in
    the compact hole-space intermediates while returning real spatial weights
    for the scalar Coulomb-integral derivatives.
    """
    ncore = Ccore.shape[1]
    nact = Cact.shape[1]
    if gamma1_act.shape != (nact, nact):
        raise ValueError(
            f"Expected active 1-RDM shape {(nact, nact)}, got {gamma1_act.shape}."
        )
    if gamma2_act.shape != (nact, nact, nact, nact):
        raise ValueError(
            "Expected active 2-RDM shape "
            f"{(nact, nact, nact, nact)}, got {gamma2_act.shape}."
        )

    Ch = np.hstack((Ccore, Cact))
    nhole = Ch.shape[1]
    gamma_h = np.zeros((nhole, nhole), dtype=np.complex128)
    gamma_h[:ncore, :ncore] = np.eye(ncore)
    gamma_h[ncore:, ncore:] = gamma1_act
    lambda2_act = make_2cumulant_so(gamma1_act, gamma2_act)

    nbf = system.nbf
    Ch_a = Ch[:nbf]
    Ch_b = Ch[nbf:]
    Z_ao = build_metric_inverted_three_center(system)
    Z_h = np.einsum("mx,Pmn,ny->Pxy", Ch_a.conj(), Z_ao, Ch_a, optimize=True)
    Z_h += np.einsum("mx,Pmn,ny->Pxy", Ch_b.conj(), Z_ao, Ch_b, optimize=True)

    W2, W3_h = _build_mc_df_hole_weights(
        gamma_h,
        lambda2_act,
        Z_h,
        ncore,
        exchange_factor=1.0,
    )

    W3 = np.einsum("mx,Pxy,ny->Pmn", Ch_a.conj(), W3_h, Ch_a, optimize=True)
    W3 += np.einsum("mx,Pxy,ny->Pmn", Ch_b.conj(), W3_h, Ch_b, optimize=True)
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
    state-specific CASSCF/GASSCF wave function, the Hermitian part of this
    matrix is the orbital Lagrange multiplier that contracts the overlap
    derivative. This helper recomputes it in the current final MO basis and
    transforms

    .. math::
        W^S_{\mu\nu}
        =
        C_{\mu p}
        \frac{1}{2}(A_{pq}+A^*_{qp})
        C^*_{\nu q}.

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
    optimizer_type = (
        RelOrbOptimizer if isinstance(mc.ci_solver, RelCIBase) else OrbOptimizer
    )
    orb_opt = getattr(mc, "orb_opt", None)
    can_reuse = (
        isinstance(orb_opt, optimizer_type)
        and hasattr(orb_opt, "Fcore")
        and hasattr(orb_opt, "eri_gaaa")
        and orb_opt.C.shape == C.shape
        and np.allclose(orb_opt.C, C, rtol=0.0, atol=1.0e-12)
    )
    if not can_reuse:
        orb_opt = optimizer_type(
            C,
            (mc.mo_space.core, mc.mo_space.actv, mc.mo_space.virt),
            mc.system.fock_builder,
            mc.system.ints_hcore(),
            mc.system.nuclear_repulsion,
            mc.nrr,
            compute_active_hessian=False,
        )
        orb_opt._compute_Fcore()
        orb_opt.get_eri_gaaa()

    orb_opt.set_rdms(gamma1_act, gamma2_act)
    lagrangian_mo = orb_opt.compute_orbital_lagrangian()
    return np.einsum("mp,pq,nq->mn", C, lagrangian_mo, C.conj(), optimize=True)


def _validate_casscf_gradient_request(mc) -> None:
    """Reject unsupported CASSCF/GASSCF options before running the method."""
    is_relativistic = isinstance(mc.ci_solver, RelCIBase)

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

    system = _find_upstream_system(mc)
    validate_df_gradient_system(system, "CASSCF/GASSCF")

    if system.two_component and not is_relativistic:
        raise NotImplementedError(
            "Two-component CASSCF/GASSCF gradients require a relativistic CI solver."
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


def _validate_converged_casscf_gradient(mc) -> None:
    """Validate restrictions that require a materialized CASSCF wave function."""
    if not mc.converged:
        raise RuntimeError(
            "CASSCF/GASSCF gradients require a converged orbital optimization."
        )

    convergence_status = mc.ci_solver.get_convergence_status()
    if convergence_status is not None and not all(convergence_status):
        raise RuntimeError(
            "CASSCF/GASSCF gradients require converged CI roots; "
            f"convergence status: {convergence_status}."
        )

    is_relativistic = isinstance(mc.ci_solver, RelCIBase)
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
    if np.iscomplexobj(mc.C[0]) and not is_relativistic:
        raise NotImplementedError(
            "Nonrelativistic CASSCF/GASSCF gradients with complex orbitals are not "
            "implemented."
        )
    if is_relativistic and mc.C[0].shape[0] != 2 * mc.system.nbf:
        raise ValueError(
            "Relativistic CASSCF/GASSCF gradients require spinor AO coefficients "
            "with 2 * system.nbf rows."
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


def _find_upstream_system(method):
    """Return the System before an unexecuted composition chain is materialized."""
    current = method
    while current is not None:
        if hasattr(current, "system"):
            return current.system
        current = getattr(current, "parent_method", None)
    raise ValueError("Could not find a System in the CASSCF parent-method chain.")
