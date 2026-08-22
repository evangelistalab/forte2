"""Coupled orbital--CI response functions for nonrelativistic SA-MCSCF.

Each routine takes the current :class:`forte2.MCOptimizer`-like object as ``mc``.
"""

import numpy as np
import scipy.sparse.linalg as spla

from forte2.lib.ci_helpers import CISigmaBuilder


def _make_working_2rdm(g2):
    r"""Return :math:`D_{tu,vw}=\tfrac12(\gamma^{tv}_{uw}+\gamma^{uv}_{tw})`."""
    return 0.5 * (np.einsum("prqs->pqrs", g2) + np.einsum("qrps->pqrs", g2))


def _validate_orbital_response_vector(orb_opt, vector):
    vector = np.asarray(vector)
    if vector.shape != (orb_opt.nrot,):
        raise ValueError(
            f"Expected an orbital-response vector with shape ({orb_opt.nrot},), "
            f"got {vector.shape}."
        )
    if np.iscomplexobj(vector):
        raise TypeError("The nonrelativistic orbital-response vector must be real.")
    return vector.astype(float, copy=False)


def _build_coupled_response_intermediates(orb_opt):
    r"""Return shared workspaces for :math:`A^{oo}`, :math:`A^{oc}`, and :math:`A^{co}`."""
    Fcore_ao = orb_opt.fock_builder.build_core_fock(orb_opt.Ccore, hcore=orb_opt.hcore)
    Fact_ao = orb_opt.fock_builder.build_active_fock(orb_opt.Cact, orb_opt.g1)
    Fcore_mo = orb_opt._transform_ao_operator(Fcore_ao, orb_opt.C)
    Fact_mo = orb_opt._transform_ao_operator(Fact_ao, orb_opt.C)
    B_ga = _transform_df_block(orb_opt, orb_opt.C, orb_opt.Cact)
    return (
        (Fcore_mo, Fact_mo, B_ga),
        (Fcore_mo, B_ga),
        (Fcore_ao, Fcore_mo, B_ga),
    )


def _transform_df_block(orb_opt, left, right):
    r"""Return :math:`B^P_{ab}=\sum_{\mu\nu}L_{\mu a}B^P_{\mu\nu}R_{\nu b}`."""
    return np.einsum(
        "Pmn,mp,nq->Ppq",
        orb_opt.fock_builder.B_Pmn,
        left,
        right,
        optimize=True,
    )


def _build_transition_fock_response(
    orb_opt,
    response_orbitals,
    reference_orbitals,
    metric=None,
    *,
    coulomb_factor,
    exchange_factor,
):
    r"""Return :math:`c_JJ[D]-c_KK[D]` for
    :math:`D=C_LG_sC_R^{\mathsf T}+C_RG_sC_L^{\mathsf T}` using signed
    low-rank factors; :math:`G_s=I` when ``metric`` is omitted.
    """
    if metric is None:
        left = response_orbitals
        right = reference_orbitals
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
        roots = np.sqrt(np.abs(eigenvalues))
        left = response_orbitals @ (eigenvectors * roots[None, :])
        right = reference_orbitals @ (
            eigenvectors * (np.sign(eigenvalues) * roots)[None, :]
        )

    density_response = left @ right.T + right @ left.T
    J_response = orb_opt.fock_builder.build_J([density_response])[0]

    if orb_opt.fock_builder.store_B_nPm:
        left_half = orb_opt.fock_builder.B_nPm @ left
        right_half = orb_opt.fock_builder.B_nPm @ right
        K_lr = np.tensordot(left_half, right_half, axes=([1, 2], [1, 2]))
    else:
        B = orb_opt.fock_builder.B_Pmn
        left_half = np.einsum("Pms,si->Pmi", B, left, optimize=True)
        right_half = np.einsum("Pms,si->Pmi", B, right, optimize=True)
        K_lr = np.einsum("Pmi,Pni->mn", left_half, right_half, optimize=True)
    K_response = K_lr + K_lr.T
    return coulomb_factor * J_response - exchange_factor * K_response


def _compute_orbital_hessian_vector_product(orb_opt, vector, intermediates):
    r"""Return :math:`[A^{oo}z]_I=2(\dot{\bar A}^{q_I}_{p_I}-\dot{\bar A}^{p_I}_{q_I})`."""
    A_response = _compute_orbital_lagrangian_response(orb_opt, vector, intermediates)
    gradient_response = 2.0 * (A_response - A_response.T)
    return orb_opt._mat_to_vec(gradient_response)


def _compute_orbital_lagrangian_response(orb_opt, vector, intermediates):
    r"""Return :math:`\dot{\bar A}[z]=\left.d\bar A[Ce^{\epsilon Z}]/d\epsilon\right|_0` at fixed RDMs."""
    Fcore_mo, Fact_mo, B_ga = intermediates
    Z = orb_opt._vec_to_mat(vector)
    C_response = orb_opt.C @ Z
    Ccore_response = C_response[:, orb_opt.core]
    Cact_response = C_response[:, orb_opt.actv]

    Fcore_ao_response = _build_transition_fock_response(
        orb_opt,
        Ccore_response,
        orb_opt.Ccore,
        coulomb_factor=2.0,
        exchange_factor=1.0,
    )
    Fact_ao_response = _build_transition_fock_response(
        orb_opt,
        Cact_response,
        orb_opt.Cact,
        orb_opt.g1,
        coulomb_factor=1.0,
        exchange_factor=0.5,
    )

    def transform_response(operator_mo, operator_ao_response):
        r"""Return :math:`\dot F=Z^{\mathsf T}F+FZ+C^{\mathsf T}\dot F^{AO}C`."""
        return (
            Z.T @ operator_mo
            + operator_mo @ Z
            + orb_opt._transform_ao_operator(operator_ao_response, orb_opt.C)
        )

    Fcore_response = transform_response(Fcore_mo, Fcore_ao_response)
    Fact_response = transform_response(Fact_mo, Fact_ao_response)

    B_response_ga = np.einsum("rp,Pru->Ppu", Z, B_ga, optimize=True)
    B_response_ga += _transform_df_block(orb_opt, orb_opt.C, Cact_response)
    B_aa = B_ga[:, orb_opt.actv, :]
    B_response_aa = B_response_ga[:, orb_opt.actv, :]

    A_response = np.zeros_like(Fcore_response)
    A_response[:, orb_opt.core] = (
        2.0 * (Fcore_response + Fact_response)[:, orb_opt.core]
    )
    A_response[:, orb_opt.actv] = np.einsum(
        "rv,vu->ru", Fcore_response[:, orb_opt.actv], orb_opt.g1, optimize=True
    )
    contracted_base = np.einsum("Pvw,tuvw->Ptu", B_aa, orb_opt.g2, optimize=True)
    contracted_response = np.einsum(
        "Pvw,tuvw->Ptu", B_response_aa, orb_opt.g2, optimize=True
    )
    A_response[:, orb_opt.actv] += np.einsum(
        "Prt,Ptu->ru", B_response_ga, contracted_base, optimize=True
    )
    A_response[:, orb_opt.actv] += np.einsum(
        "Prt,Ptu->ru", B_ga, contracted_response, optimize=True
    )
    return A_response


def _compute_active_space_hamiltonian_response(orb_opt, vector, intermediates):
    r"""Return :math:`\hat H[z]=\dot E_C+\dot F_C{}^u_v\hat E^u_v+
    \tfrac12\dot{\langle uv|tw\rangle}\hat E^{uv}_{tw}`.
    """
    Fcore_ao, Fcore_mo, B_ga = intermediates
    Z = orb_opt._vec_to_mat(vector)
    C_response = orb_opt.C @ Z
    Ccore_response = C_response[:, orb_opt.core]

    Fcore_ao_response = _build_transition_fock_response(
        orb_opt,
        Ccore_response,
        orb_opt.Ccore,
        coulomb_factor=2.0,
        exchange_factor=1.0,
    )
    Fcore_response = (
        Z.T @ Fcore_mo
        + Fcore_mo @ Z
        + orb_opt._transform_ao_operator(Fcore_ao_response, orb_opt.C)
    )

    h_plus_fcore = orb_opt.hcore + Fcore_ao
    scalar_response = np.trace(
        Ccore_response.T @ h_plus_fcore @ orb_opt.Ccore
        + orb_opt.Ccore.T @ h_plus_fcore @ Ccore_response
        + orb_opt.Ccore.T @ Fcore_ao_response @ orb_opt.Ccore
    )

    B_active = B_ga[:, orb_opt.actv, :]
    B_left_response = np.einsum("ru,Prv->Puv", Z[:, orb_opt.actv], B_ga, optimize=True)
    B_active_response = B_left_response + B_left_response.transpose(0, 2, 1)
    eri_response = np.einsum(
        "Put,Pvw->uvtw", B_active_response, B_active, optimize=True
    )
    eri_response += np.einsum(
        "Put,Pvw->uvtw", B_active, B_active_response, optimize=True
    )

    one_body_response = Fcore_response[orb_opt.actv, orb_opt.actv]
    return (
        float(scalar_response),
        np.ascontiguousarray(one_body_response),
        np.ascontiguousarray(eri_response),
    )


def _build_orbital_lagrangian_from_rdms(
    orb_opt,
    overlap_response,
    g1_response,
    g2_response,
    intermediates,
):
    r"""Return the MO orbital Lagrangian :math:`A[s,\gamma,D]`."""
    return _build_orbital_lagrangian(
        orb_opt,
        overlap_response,
        g1_response,
        _make_working_2rdm(g2_response),
        intermediates,
    )


def _build_orbital_lagrangian(
    orb_opt,
    overlap_response,
    g1_response,
    g2_working_response,
    intermediates,
):
    r"""Return :math:`A[s,\gamma,D]` from a working-form 2-RDM:

    .. math::

        A^m_p&=2(sF_{C,p}^m+F_{A,p}^m[\gamma]),\\
        A^u_p&=\sum_vF_{C,p}^v\gamma^v_u
        +\sum_{tvw}\langle pv|tw\rangle D_{tu,vw},\\
        A^e_p&=0.
    """
    Fcore_mo, B_ga = intermediates
    Fact_ao_response = _build_transition_fock_response(
        orb_opt,
        orb_opt.Cact,
        orb_opt.Cact,
        0.5 * g1_response,
        coulomb_factor=1.0,
        exchange_factor=0.5,
    )
    Fact_response = orb_opt._transform_ao_operator(Fact_ao_response, orb_opt.C)

    A_response = np.zeros_like(Fcore_mo)
    A_response[:, orb_opt.core] = (
        2.0 * (overlap_response * Fcore_mo + Fact_response)[:, orb_opt.core]
    )
    A_response[:, orb_opt.actv] = np.einsum(
        "rv,vu->ru", Fcore_mo[:, orb_opt.actv], g1_response, optimize=True
    )
    B_aa = B_ga[:, orb_opt.actv, :]
    contracted_density = np.einsum(
        "Pvw,tuvw->Ptu", B_aa, g2_working_response, optimize=True
    )
    A_response[:, orb_opt.actv] += np.einsum(
        "Prt,Ptu->ru", B_ga, contracted_density, optimize=True
    )
    return A_response


def _compute_ci_orbital_response_from_rdms(
    orb_opt,
    overlap_response,
    g1_response,
    g2_response,
    intermediates,
):
    r"""Return :math:`[A^{oc}x]_I=2(A[x]^{q_I}_{p_I}-A[x]^{p_I}_{q_I})`."""
    A_response = _build_orbital_lagrangian_from_rdms(
        orb_opt,
        overlap_response,
        g1_response,
        g2_response,
        intermediates,
    )
    gradient_response = 2.0 * (A_response - A_response.T)
    return orb_opt._mat_to_vec(gradient_response)


def _get_ci_response_layout(mc):
    r"""Return root-major slices and :math:`n_{CI}=\sum_\alpha n_\alpha`."""
    layout = []
    start = 0
    for absolute_root, (state_index, root_in_state) in enumerate(
        mc.ci_solver.sa_info.absolute_root_map
    ):
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        stop = start + sub_solver.basis_size
        layout.append(
            (
                absolute_root,
                state_index,
                root_in_state,
                slice(start, stop),
            )
        )
        start = stop
    return tuple(layout), start


def get_ci_response_layout(mc):
    r"""Return the root-major CI coefficient layout.

    .. math::

        \mathbf x
        =\bigoplus_{\alpha=0}^{d-1}\mathbf x_\alpha,
        \qquad x_J=(x_\alpha)_M
        \quad\text{for }J\leftrightarrow(\alpha,M)\in\mathbb L.

    Returns
    -------
    tuple[tuple[int, int, int, slice], ...]
        ``(absolute_root, state_index, root_in_state, slice)`` entries.
    """
    layout, _ = _get_ci_response_layout(mc)
    return layout


def _resolve_absolute_root(mc, root, *, allow_single_root_default=False):
    """Resolve an absolute state-average root index."""
    nroots = mc.ci_solver.sa_info.nroots_sum
    if root is None and allow_single_root_default:
        if nroots == 1:
            return 0
        raise ValueError(
            "An absolute root must be specified when more than one root is present."
        )
    if isinstance(root, bool) or not isinstance(root, (int, np.integer)):
        raise TypeError("The target root must be an integer.")
    if root < 0 or root >= nroots:
        raise ValueError(f"Expected a target root in [0, {nroots}), got {root}.")
    return int(root)


def _validate_ci_response_vector(mc, ci_vector):
    layout, nci = _get_ci_response_layout(mc)
    ci_vector = np.asarray(ci_vector)
    if ci_vector.shape != (nci,):
        raise ValueError(
            f"Expected a CI response vector with shape ({nci},), "
            f"got {ci_vector.shape}."
        )
    if np.iscomplexobj(ci_vector):
        raise TypeError("The nonrelativistic orbital--CI response vector must be real.")
    return ci_vector.astype(float, copy=False), layout


def _project_ci_response_vector(mc, ci_vector, layout):
    r"""Return :math:`Q_sx_\alpha=(I_s-\sum_{\gamma\in R_s}c_\gamma c_\gamma^T)x_\alpha`."""
    projected = np.empty_like(ci_vector)
    for _, state_index, _, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        solved_roots = sub_solver.evecs[:, : sub_solver.nroot]
        root_vector = ci_vector[coefficient_slice]
        projected[coefficient_slice] = root_vector - solved_roots @ (
            solved_roots.T @ root_vector
        )
    return projected


def _build_ci_reference_det_vectors(mc, layout):
    r"""Return fixed references
    :math:`c_{\alpha,I}^{\mathrm{det}}=\sum_M T^\alpha_{IM}c_{\alpha,M}^{\mathrm{CSF}}`.
    """
    return tuple(
        mc.ci_solver.sub_solvers[state_index].csf_C_to_det_C(
            mc.ci_solver.sub_solvers[state_index].evecs[:, root_in_state]
        )
        for _, state_index, root_in_state, _ in layout
    )


def _compute_ci_response_rdms(mc, ci_vector, layout, reference_dets=None):
    r"""Return the root sums

    .. math::

        s[\mathbf x]&=\sum_\alpha
        (\mathbf x_\alpha^T\mathbf c_\alpha
        +\mathbf c_\alpha^T\mathbf x_\alpha),\\
        \gamma[\mathbf x]&=\sum_\alpha
        (\gamma[Tx_\alpha,Tc_\alpha]+\gamma[Tc_\alpha,Tx_\alpha]),\\
        \Gamma[\mathbf x]&=\sum_\alpha
        (\Gamma[Tx_\alpha,Tc_\alpha]+\Gamma[Tc_\alpha,Tx_\alpha]).
    """
    nact = mc.mo_space.nactv
    overlap_response = 0.0
    g1_response = np.zeros((nact,) * 2, dtype=float)
    g2_response = np.zeros((nact,) * 4, dtype=float)

    for index, (_, state_index, root_in_state, coefficient_slice) in enumerate(layout):
        response = ci_vector[coefficient_slice]
        if not np.any(response):
            continue

        sub_solver = mc.ci_solver.sub_solvers[state_index]
        reference = sub_solver.evecs[:, root_in_state]
        overlap_response += np.dot(response, reference)
        overlap_response += np.dot(reference, response)

        response_det = sub_solver.csf_C_to_det_C(response)
        reference_det = (
            sub_solver.csf_C_to_det_C(reference)
            if reference_dets is None
            else reference_dets[index]
        )
        sigma_builder = sub_solver.ci_sigma_builder
        g1_response += sigma_builder.sf_1rdm(response_det, reference_det)
        g1_response += sigma_builder.sf_1rdm(reference_det, response_det)
        g2_response += sigma_builder.sf_2rdm(response_det, reference_det)
        g2_response += sigma_builder.sf_2rdm(reference_det, response_det)

    return overlap_response, g1_response, g2_response


def _compute_ci_orbital_hessian_vector_product(
    mc, orbital_vector, layout, intermediates
):
    r"""Return :math:`[A^{co}z]_\alpha=2w_\alpha\hat H[z]c_\alpha`."""
    scalar_response, one_body_response, two_body_response = (
        _compute_active_space_hamiltonian_response(
            mc.orb_opt, orbital_vector, intermediates
        )
    )
    response = np.empty(layout[-1][-1].stop, dtype=float)
    builders = {}

    for absolute_root, state_index, root_in_state, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        if state_index not in builders:
            builder = CISigmaBuilder(
                sub_solver.ci_strings,
                scalar_response,
                one_body_response,
                two_body_response,
                sub_solver.log_level,
            )
            builder.set_memory(sub_solver.ci_params.ci_builder_memory)
            algorithm = sub_solver.ci_params.ci_algorithm.lower()
            builder.set_algorithm("kh" if algorithm == "exact" else algorithm)
            builders[state_index] = builder

        reference = sub_solver.evecs[:, root_in_state]
        sigma_csf = _apply_ci_hamiltonian_to_csf(
            sub_solver, builders[state_index], reference
        )
        weight = mc.ci_solver.weights_flat[absolute_root]
        response[coefficient_slice] = 2.0 * weight * sigma_csf

    return response


def _apply_ci_hamiltonian_to_csf(sub_solver, sigma_builder, vector):
    r"""Return :math:`\sigma_{CSF}=T^{\mathsf T}H_{det}Tc_{CSF}`."""
    vector_det = sub_solver.csf_C_to_det_C(vector)
    sigma_det = np.empty(sub_solver.ndet, dtype=float)
    sigma_builder.Hamiltonian(vector_det, sigma_det)
    sigma_csf = np.empty(sub_solver.basis_size, dtype=float)
    sub_solver.spin_adapter.det_C_to_csf_C(sigma_det, sigma_csf)
    return sigma_csf


def _compute_ci_ci_hessian_vector_product(mc, ci_vector, layout):
    r"""Return :math:`[A^{cc}x]_\alpha=2(H_\alpha-E_\alpha I)x_\alpha`."""
    response = np.empty_like(ci_vector)
    for absolute_root, state_index, _, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        root_vector = ci_vector[coefficient_slice]
        sigma = _apply_ci_hamiltonian_to_csf(
            sub_solver, sub_solver.ci_sigma_builder, root_vector
        )
        response[coefficient_slice] = 2.0 * (
            sigma - mc.E_ci[absolute_root] * root_vector
        )
    return response


def _compute_raw_ci_response_b_vector(mc, root, layout):
    r"""Return :math:`(\widetilde b^c_\alpha)_\beta=2\delta_{\alpha\beta}H_\beta c_\beta`."""
    _, state_index, root_in_state, coefficient_slice = layout[root]
    sub_solver = mc.ci_solver.sub_solvers[state_index]
    reference = sub_solver.evecs[:, root_in_state]
    sigma = _apply_ci_hamiltonian_to_csf(
        sub_solver, sub_solver.ci_sigma_builder, reference
    )
    raw_b = np.zeros(layout[-1][-1].stop, dtype=float)
    raw_b[coefficient_slice] = 2.0 * sigma
    return raw_b


def compute_projected_response_vector_product(mc, orbital_vector, ci_vector):
    r"""Apply the gauge-fixed projected coupled response operator.

    .. math::

        \binom{\mathbf y^{\mathrm o}}{\mathbf y^{\mathrm c}}
        =\begin{pmatrix}
         \mathcal A^{\mathrm{oo}}&\mathcal A^{\mathrm{oc}}Q\\
         Q\mathcal A^{\mathrm{co}}&Q\mathcal A^{\mathrm{cc}}Q+P
         \end{pmatrix}\binom{\mathbf z}{\mathbf x},\qquad P=I-Q.

    Parameters
    ----------
    orbital_vector : np.ndarray
        Real nonredundant orbital vector with shape ``(nrot,)``.
    ci_vector : np.ndarray
        Root-major real CI vector with shape ``(nci,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Orbital and gauge-fixed projected CI products.
    """
    orbital_vector = _validate_orbital_response_vector(mc.orb_opt, orbital_vector)
    ci_vector, layout = _validate_ci_response_vector(mc, ci_vector)
    intermediates = _build_coupled_response_intermediates(
        mc.orb_opt,
    )
    return _compute_projected_response_vector_product(
        mc, orbital_vector, ci_vector, layout, intermediates
    )


def _compute_projected_response_vector_product(
    mc,
    orbital_vector,
    ci_vector,
    layout,
    intermediates,
    reference_dets=None,
):
    r"""Return :math:`y^o=A_{oo}z+A_{oc}Qx` and
    :math:`y^c=Q(A_{co}z+A_{cc}Qx)+Px` using shared intermediates.
    """
    projected_ci = _project_ci_response_vector(mc, ci_vector, layout)

    orbital_intermediates, density_intermediates, hamiltonian_intermediates = (
        intermediates
    )
    orbital_product = _compute_orbital_hessian_vector_product(
        mc.orb_opt, orbital_vector, orbital_intermediates
    )
    density_response = _compute_ci_response_rdms(
        mc, projected_ci, layout, reference_dets
    )
    orbital_product += _compute_ci_orbital_response_from_rdms(
        mc.orb_opt, *density_response, density_intermediates
    )

    raw_ci_product = _compute_ci_orbital_hessian_vector_product(
        mc, orbital_vector, layout, hamiltonian_intermediates
    )
    raw_ci_product += _compute_ci_ci_hessian_vector_product(mc, projected_ci, layout)
    ci_product = _project_ci_response_vector(mc, raw_ci_product, layout)
    ci_product += ci_vector - projected_ci
    return orbital_product, ci_product


def _invert_response_diagonal(diagonal):
    r"""Return :math:`1/\widetilde d_i` after sign-preserving :math:`10^{-6}\|d\|_\infty` flooring."""
    diagonal = np.asarray(diagonal, dtype=float)
    scale = max(1.0, float(np.max(np.abs(diagonal), initial=0.0)))
    floor = 1.0e-6 * scale
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    regularized = np.where(np.abs(diagonal) < floor, signs * floor, diagonal)
    return 1.0 / regularized


def _build_response_preconditioner(mc, layout, nrot, nci):
    r"""Return the projected block-Jacobi :math:`M^{-1}` used by GMRES."""
    orbital_diagonal = mc.orb_opt._mat_to_vec(mc.orb_opt._compute_orbhess())
    orbital_inverse = _invert_response_diagonal(orbital_diagonal)

    ci_inverse = np.empty(nci, dtype=float)
    state_diagonals = {}
    for absolute_root, state_index, _, coefficient_slice in layout:
        sub_solver = mc.ci_solver.sub_solvers[state_index]
        if state_index not in state_diagonals:
            state_diagonals[state_index] = sub_solver.ci_sigma_builder.form_Hdiag_csf(
                sub_solver.dets,
                sub_solver.spin_adapter,
                False,
            )
        diagonal = 2.0 * (state_diagonals[state_index] - mc.E_ci[absolute_root])
        ci_inverse[coefficient_slice] = _invert_response_diagonal(diagonal)

    dimension = nrot + nci

    def matvec(vector):
        r"""Apply :math:`\mathbf y=\mathcal M^{-1}\mathbf v`."""
        product = np.empty(dimension, dtype=float)
        product[:nrot] = orbital_inverse * vector[:nrot]
        ci_vector = vector[nrot:]
        projected_ci = _project_ci_response_vector(mc, ci_vector, layout)
        product[nrot:] = _project_ci_response_vector(
            mc, ci_inverse * projected_ci, layout
        )
        product[nrot:] += ci_vector - projected_ci
        return product

    return spla.LinearOperator((dimension, dimension), matvec=matvec, dtype=float)


def _solve_response(mc, root, *, r_tol, maxiter):
    r"""Return :math:`(\mathbf z_\alpha,\mathbf x_\alpha)` satisfying
    :math:`\mathscr A(\mathbf z_\alpha,\mathbf x_\alpha)^T
    =-(\mathbf b^o_\alpha,\mathbf b^c_\alpha)^T`, plus its workspace.
    """
    root = _resolve_absolute_root(mc, root)
    if not np.isscalar(r_tol) or r_tol <= 0.0:
        raise ValueError(f"r_tol must be positive, got {r_tol}.")
    if maxiter is not None and (
        isinstance(maxiter, bool)
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 1
    ):
        raise ValueError(f"maxiter must be a positive integer, got {maxiter}.")

    layout, nci = _get_ci_response_layout(mc)
    reference_dets = _build_ci_reference_det_vectors(mc, layout)
    nrot = mc.orb_opt.nrot
    dimension = nrot + nci
    mc.orb_opt.set_rdms(mc.make_average_1rdm(), mc.make_average_2rdm())
    intermediates = _build_coupled_response_intermediates(mc.orb_opt)
    _, density_intermediates, _ = intermediates
    orbital_b = _compute_ci_orbital_response_from_rdms(
        mc.orb_opt,
        1.0,
        mc.make_sf_1rdm(root),
        mc.make_sf_2rdm(root),
        density_intermediates,
    )
    raw_ci_b = _compute_raw_ci_response_b_vector(mc, root, layout)
    ci_b = _project_ci_response_vector(mc, raw_ci_b, layout)
    rhs = -np.concatenate((orbital_b, ci_b))

    def matvec(vector):
        r"""Apply :math:`\mathbf y=\mathscr A\mathbf v`."""
        orbital_product, ci_product = _compute_projected_response_vector_product(
            mc,
            vector[:nrot],
            vector[nrot:],
            layout,
            intermediates,
            reference_dets,
        )
        product = np.empty(dimension, dtype=float)
        product[:nrot] = orbital_product
        product[nrot:] = ci_product
        return product

    operator = spla.LinearOperator((dimension, dimension), matvec=matvec, dtype=float)
    preconditioner = _build_response_preconditioner(mc, layout, nrot, nci)
    solution, info = spla.gmres(
        operator,
        rhs,
        rtol=float(r_tol),
        atol=0.0,
        restart=min(dimension, 50),
        maxiter=maxiter,
        M=preconditioner,
    )
    if info != 0:
        reason = (
            f"did not converge after {info} iterations"
            if info > 0
            else f"failed with status {info}"
        )
        raise RuntimeError(f"The coupled CASSCF response solve {reason}.")

    orbital_response = solution[:nrot]
    ci_response = _project_ci_response_vector(mc, solution[nrot:], layout)
    return (
        orbital_response,
        ci_response,
        root,
        layout,
        intermediates,
        reference_dets,
    )


def solve_state_specific_response(
    mc,
    root,
    *,
    r_tol=1.0e-10,
    maxiter=None,
):
    r"""Solve the projected coupled response equations for one target root.

    .. math::

        \begin{pmatrix}
         \mathcal A^{\mathrm{oo}} & \mathcal A^{\mathrm{oc}}Q\\
         Q\mathcal A^{\mathrm{co}} &
         Q\mathcal A^{\mathrm{cc}}Q+P
        \end{pmatrix}
        \begin{pmatrix}\mathbf z_\alpha\\\mathbf x_\alpha\end{pmatrix}
        =-
        \begin{pmatrix}
         \mathbf b^{\mathrm o}_\alpha\\
         \mathbf b^{\mathrm c}_\alpha
        \end{pmatrix},

    Here :math:`P=I-Q`.  GMRES uses a block-diagonal preconditioner and
    never forms a dense response block or CI-to-RDM Jacobian.

    Parameters
    ----------
    root : int
        Absolute target-root index in state-average ordering.
    r_tol : float, optional
        Relative GMRES residual tolerance.
    maxiter : int or None, optional
        Maximum number of GMRES restart cycles.  The SciPy default is used
        when omitted.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Orbital and projected root-major CI responses.

    Raises
    ------
    RuntimeError
        If GMRES does not converge.
    """
    orbital_response, ci_response, _, _, _, _ = _solve_response(
        mc, root, r_tol=r_tol, maxiter=maxiter
    )
    return orbital_response, ci_response


def _compute_omega_from_intermediates(
    mc,
    orbital_response,
    overlap_response,
    g1_response,
    g2_response,
    intermediates,
):
    r"""Return the relaxed multiplier from combined target/CI RDMs:

    .. math::

        \Omega_\alpha
        &=A[1+s[\mathbf x_\alpha],
             \gamma^\alpha+\gamma[\mathbf x_\alpha],
             \Gamma^\alpha+\Gamma[\mathbf x_\alpha]]
          +\dot{\bar A}[\mathbf z_\alpha]
          +Z_\alpha\bar A-\bar A Z_\alpha,\\
        \omega_\alpha&=\tfrac12(\Omega_\alpha+\Omega_\alpha^T).
    """
    orbital_intermediates, density_intermediates, _ = intermediates
    base_A = _build_orbital_lagrangian_from_rdms(
        mc.orb_opt,
        overlap_response,
        g1_response,
        g2_response,
        density_intermediates,
    )
    average_A = _build_orbital_lagrangian(
        mc.orb_opt,
        1.0,
        mc.orb_opt.g1,
        mc.orb_opt.g2,
        density_intermediates,
    )
    orbital_A = _compute_orbital_lagrangian_response(
        mc.orb_opt, orbital_response, orbital_intermediates
    )
    Z = mc.orb_opt._vec_to_mat(orbital_response)
    orbital_A += Z @ average_A - average_A @ Z
    Omega = base_A + orbital_A
    return 0.5 * (Omega + Omega.T)


def _solve_state_specific_gradient_response(mc, root):
    r"""Return
    :math:`(\mathbf z_\alpha,\omega_\alpha,
    \gamma^\alpha+\gamma[\mathbf x_\alpha],
    \Gamma^\alpha+\Gamma[\mathbf x_\alpha])`.
    """
    (
        orbital_response,
        ci_response,
        root,
        layout,
        intermediates,
        reference_dets,
    ) = _solve_response(mc, root, r_tol=1.0e-10, maxiter=None)
    ci_overlap, ci_g1, ci_g2 = _compute_ci_response_rdms(
        mc, ci_response, layout, reference_dets
    )
    del ci_response

    base_g1 = mc.make_sf_1rdm(root)
    base_g2 = mc.make_sf_2rdm(root)
    base_g1 += ci_g1
    base_g2 += ci_g2
    del ci_g1, ci_g2

    omega = _compute_omega_from_intermediates(
        mc,
        orbital_response,
        1.0 + ci_overlap,
        base_g1,
        base_g2,
        intermediates,
    )
    return orbital_response, omega, base_g1, base_g2


def compute_omega(mc, root, orbital_response, ci_response):
    r"""Compute the relaxed MO orthogonality multiplier for one root.

    For target root :math:`\alpha`, let :math:`A^\alpha` be its orbital
    Lagrangian, :math:`A^{\mathrm{oc}}[\mathbf x_\alpha]` the contribution
    from the root-major CI multiplier, and :math:`\dot{\bar A}[\mathbf
    z_\alpha]` the directional response of the state-averaged orbital
    Lagrangian.  In the current real nonrelativistic convention this method
    forms

    .. math::

        \Omega_\alpha
        = A^\alpha
        + A^{\mathrm{oc}}[\mathbf x_\alpha]
        + \dot{\bar A}[\mathbf z_\alpha]
        + Z_\alpha\bar A-\bar A Z_\alpha,

        \omega_\alpha
        = \frac{1}{2}
          (\Omega_\alpha+\Omega_\alpha^{\mathsf T}).

    The commutator accounts for the moving orbital frame.  The returned
    matrix is in the current MO basis.

    Parameters
    ----------
    root : int
        Absolute target-root index in state-average ordering.
    orbital_response : np.ndarray
        Solved nonredundant orbital multiplier ``z_alpha``.
    ci_response : np.ndarray
        Solved root-major CI multiplier ``x_alpha``.

    Returns
    -------
    np.ndarray
        Real symmetric ``omega_alpha`` with shape ``(nmo, nmo)`` in the
        current MO basis.
    """
    root = _resolve_absolute_root(mc, root)
    orbital_response = _validate_orbital_response_vector(mc.orb_opt, orbital_response)
    ci_response, layout = _validate_ci_response_vector(mc, ci_response)
    ci_response = _project_ci_response_vector(mc, ci_response, layout)
    ci_overlap, ci_g1, ci_g2 = _compute_ci_response_rdms(mc, ci_response, layout)
    base_g1 = mc.make_sf_1rdm(root)
    base_g2 = mc.make_sf_2rdm(root)
    base_g1 += ci_g1
    base_g2 += ci_g2

    mc.orb_opt.set_rdms(mc.make_average_1rdm(), mc.make_average_2rdm())
    intermediates = _build_coupled_response_intermediates(mc.orb_opt)
    return _compute_omega_from_intermediates(
        mc,
        orbital_response,
        1.0 + ci_overlap,
        base_g1,
        base_g2,
        intermediates,
    )
