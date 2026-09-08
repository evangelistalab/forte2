Obtaining RDMs and cumulants
============================

To obtain reduced density matrices (RDMs) and their cumulants, Forte2 implements
``make_rdm`` and ``make_cumulant`` methods on all CI and MCSCF classes.

Both methods take a ``spin_type`` argument that selects the representation:

``sd``
    Spin-dependent. Each spin case comes back separately, as a tuple.
``sf``
    Spin-free. The definition of "spin-free" differs for RDMs and cumulants. See below.
``so``
    Spin-orbital. A single tensor over spinors, used by the two-component solvers.

The spelled-out forms are also accepted: ``spin_dependent`` or ``spin-dependent``,
``spin_free`` or ``spin-free``, and ``spin_orbital``, ``spin-orbital``, or ``spinorbital``.

Which orders and spin types are available depends on the solver. ``MCOptimizer`` forwards both
methods to its ``ci_solver``, so the tables that follow also describe an MCSCF calculation built
on a given solver.

.. note::
    For RDMs, "spin-free" simply means spin-summed, e.g., the spin-free three-body RDMs are given by:

    .. math::

        \Gamma^{uvw}_{xyz} = 2\left(
        {\gamma}^{u_\uparrow v_\uparrow w_\uparrow}_{x_\uparrow y_\uparrow z_\uparrow}
        + {\gamma}^{u_\uparrow v_\uparrow w_\downarrow}_{x_\uparrow y_\uparrow z_\downarrow}
        + {\gamma}^{u_\uparrow w_\uparrow v_\downarrow}_{x_\uparrow z_\uparrow y_\downarrow}
        + {\gamma}^{v_\uparrow w_\uparrow u_\downarrow}_{y_\uparrow z_\uparrow x_\downarrow}
        \right).

    For cumulants, spin-summing spin-dependent cumulants results in quantities that depend on the :math:`M_s` of the state [1]_. Instead, "spin-free" cumulants in Forte2 always means the :math:`M_s`-invariant form of the cumulant, which is defined for an equally weighted ensemble over all spin states of a multiplet [2]_. The two-body spin-free cumulants are defined by

    .. math::

        \Lambda^{uv}_{xy} = \Gamma^{uv}_{xy}
        - \Gamma^{u}_{x} \Gamma^{v}_{y}
        + \frac{1}{2} \Gamma^{u}_{y} \Gamma^{v}_{x},

    and the three-body spin-free cumulants by

    .. math::

        \begin{aligned}
        \Lambda^{uvw}_{xyz} = {} & \Gamma^{uvw}_{xyz}
        - \left(
        \Gamma^{u}_{x} \Gamma^{vw}_{yz}
        + \Gamma^{v}_{y} \Gamma^{uw}_{xz}
        + \Gamma^{w}_{z} \Gamma^{uv}_{xy}
        \right) \\
        & + \frac{1}{2} \left(
        \Gamma^{u}_{y} \Gamma^{vw}_{xz}
        + \Gamma^{u}_{z} \Gamma^{vw}_{yx}
        + \Gamma^{v}_{x} \Gamma^{uw}_{yz}
        + \Gamma^{v}_{z} \Gamma^{uw}_{xy}
        + \Gamma^{w}_{x} \Gamma^{uv}_{zy}
        + \Gamma^{w}_{y} \Gamma^{uv}_{xz}
        \right) \\
        & + 2\, \Gamma^{u}_{x} \Gamma^{v}_{y} \Gamma^{w}_{z}
        - \left(
        \Gamma^{u}_{x} \Gamma^{v}_{z} \Gamma^{w}_{y}
        + \Gamma^{u}_{y} \Gamma^{v}_{x} \Gamma^{w}_{z}
        + \Gamma^{u}_{z} \Gamma^{v}_{y} \Gamma^{w}_{x}
        \right) \\
        & + \frac{1}{2} \left(
        \Gamma^{u}_{y} \Gamma^{v}_{z} \Gamma^{w}_{x}
        + \Gamma^{u}_{z} \Gamma^{v}_{x} \Gamma^{w}_{y}
        \right).
        \end{aligned}

RDM capabilities
----------------

Pass two roots to ``make_rdm`` to get a transition RDM instead of a state RDM. Roots that belong
to the same ``State`` support transition RDMs at every order the solver implements; roots that
belong to different states are more restricted.

.. list-table:: RDM capabilities of the CI solvers
   :header-rows: 1
   :widths: 24 12 16 24 24

   * - Solver
     - Orders
     - Spin types
     - Transition RDMs within a state
     - Transition RDMs between states
   * - ``CI``
     - 1, 2, 3
     - ``sd``, ``sf``
     - All orders
     - Order 1
   * - ``RelCI``
     - 1, 2, 3
     - ``so``
     - All orders
     - Not supported
   * - ``SelectedCI``
     - 1, 2
     - ``sd``, ``sf``
     - All orders
     - Order 1
   * - ``RelSelectedCI``
     - 1, 2
     - ``so``
     - All orders
     - Not supported

A transition RDM between two states also requires the two states to have the same number of alpha
and beta electrons.
State-averaged RDMs can be obtained by ``make_average_rdm(order)``. 
There, the spin type is automatically deduced: if a non-relativistic solver is used, the averaged spin-free RDM is returned, otherwise the averaged spin-orbital RDM is returned.

Cumulant capabilities
---------------------

``make_cumulant`` takes a single root. A cumulant is defined for one state, so there is no
transition counterpart.

.. list-table:: Cumulant capabilities of the CI solvers
   :header-rows: 1
   :widths: 34 33 33

   * - Solver
     - Orders
     - Spin types
   * - ``CI``
     - 2, 3
     - ``sf``
   * - ``RelCI``
     - 2, 3
     - ``so``
   * - ``SelectedCI``
     - 2
     - ``sf``
   * - ``RelSelectedCI``
     - 2
     - ``so``

To get a state-averaged cumulant, call ``make_average_cumulant(order)`` instead. Cumulants are
nonlinear in the RDMs, so a state-averaged cumulant is built from the state-averaged RDMs rather
than from a weighted sum of per-root cumulants. 
The same deduction of spin-type as ``make_average_rdm`` is in effect.

References
----------
.. [1] Shamasundar, K. R. Cumulant Decomposition of Reduced Density Matrices, Multireference Normal Ordering, and Wick's Theorem: A Spin-Free Approach. J. Chem. Phys. 2009, 131 (17), 174109. https://doi.org/10.1063/1.3256237.
.. [2] Li, C.; Evangelista, F. A. Spin-Free Formulation of the Multireference Driven Similarity Renormalization Group: A Benchmark Study of First-Row Diatomic Molecules and Spin-Crossover Energetics. J. Chem. Phys. 2021, 155 (11), 114111. https://doi.org/10.1063/5.0059362.
