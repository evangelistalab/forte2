Projector-penalized orbital optimization
======================================

``MCOptimizer(..., lambda_penalty=...)`` can discourage the selected active
orbital subspace from drifting during CASSCF or GASSCF optimization. The default
``lambda_penalty=0.0`` gives ordinary orbital optimization. A positive, finite
value introduces a soft constraint: it favors the starting active subspace but
does not freeze the orbitals or impose exact equality with that space.

Objective and reference
-----------------------

Let :math:`C_0` contain the orthonormal starting orbitals in the optimizer's
contiguous core/active/virtual order, and write the current orbitals as
:math:`C=C_0U`. In this orthonormal initial MO frame, define

.. math::

   D_{pq}=d_p\delta_{pq},\qquad
   d_p=\begin{cases}1 & p\text{ active},\\0 & \text{otherwise},\end{cases}
   \qquad P(U)=UDU^\dagger.

The augmented objective is

.. math::

   F(U,\mathbf c)
   = E(U,\mathbf c) + f(U),\qquad
   f(U)=\lambda\lVert P(U)-D\rVert_F^2
        =2\lambda\left[n_{\mathrm{act}}-\operatorname{Tr}(DP(U))\right].

Here :math:`E` is the physical CASSCF energy, or its state-weighted average,
and :math:`\mathbf c` denotes the CI coefficients. The penalty is independent
of the CI coefficients, so their optimization still uses the physical
active-space Hamiltonian. ``lambda_penalty`` has units of Hartree.

The reference is the active subspace supplied at the start of each run. It
remains fixed throughout all orbital and CI iterations of that run. For GAS,
:math:`D` selects the union of all active orbital blocks; the penalty does not
separately constrain each GAS block. Rotations within this total active space
leave the penalty unchanged, including the final natural-orbital or
semicanonical transformation.

The penalty strength is fixed during a run. There is no automatic annealing or
release step. A subsequent unpenalized optimization is a separate calculation;
changing the option on an already converged object does not reoptimize its
wave function.

Orbital derivatives
-------------------

Orbital steps are right increments,
:math:`U\leftarrow U\exp(\kappa)`, with
:math:`\kappa^\dagger=-\kappa`. Thus the gradient used for the next step must
be in the current MO frame. With :math:`B=U^\dagger D U`, the penalty contribution
in the optimizer's packed rotation convention is

.. math::

   g^{\mathrm{pen}}_{pq}
      =4\lambda[D,B]_{pq}
      =4\lambda(d_p-d_q)B_{pq}.

For a real rotation coordinate mixing :math:`p` and :math:`q`, its exact
diagonal curvature at the current orbitals is

.. math::

   h^{\mathrm{pen}}_{pq,pq}
      =4\lambda(d_p-d_q)(B_{pp}-B_{qq}).

The same diagonal curvature applies to the imaginary rotation coordinate in
the complex formulation. At the reference, an active--external rotation has
curvature :math:`4\lambda`; away from the reference it need not retain that
value. The physical orbital Hessian remains approximate. Shared objective,
gradient, and Hessian wrappers add the projector contributions for both real
orbitals and complex spinors.

Energies and nuclear gradients
------------------------------

The reported energies distinguish the physical model from the optimization
penalty:

* ``E``, ``E_avg``, and ``E_ci`` retain their physical-energy meanings.
* ``E_penalty`` is :math:`f(U)` at the final active subspace.
* ``E_objective`` is :math:`E+E_{\mathrm{penalty}}`.

A small augmented orbital gradient establishes stationarity of :math:`F`,
not of the physical energy alone. Existing analytic CASSCF/GASSCF nuclear
gradients do not include the penalty and reference-orbital response, so a
nonzero-penalty request raises ``NotImplementedError``. This also prevents
analytic geometry optimization from using those unsupported gradients.
Merely changing ``lambda_penalty`` to zero after a penalized run does not
remove this restriction; the unpenalized wave function must first be
reoptimized.
