.. _hf-symmetry-design:

HF symmetry and occupation constraints
======================================

This note describes the implementation introduced for
`Issue #265 <https://github.com/evangelistalab/forte2/issues/265>`_:
reliable MO symmetry assignment at stretched geometries, occupation constraints
for all HF methods, and full double-group symmetry for spin-orbit GHF.
The user-facing options and examples are in :doc:`../guide/basics`.

Scope and public contract
-------------------------

``target_symmetry`` denotes the total electronic determinant's irrep. Molecular
point-group detection remains a property of ``System(symmetry=True)``.
The spatial groups are C1, Ci, C2, Cs, C2h, D2, C2v, and D2h. With spin-orbit
coupling, GHF uses the corresponding double group, rather than assigning
ordinary spatial irreps to spinors. These are the double groups of the detected
subgroup; this change does not add detection of larger molecular groups.

``irrep_occupations`` instead specifies a complete orbital configuration:

.. list-table:: Occupation semantics
   :header-rows: 1
   :widths: 20 45 35

   * - Method
     - Value for each irrep
     - Required total
   * - RHF
     - Occupied spatial orbitals, each doubly occupied
     - Number of electron pairs
   * - ROHF, UHF, CUHF
     - ``(alpha, beta)`` spatial-orbital counts
     - ``(na, nb)``
   * - GHF
     - Occupied spinors, each singly occupied
     - Number of electrons

Keys may be case-insensitive labels or integer indices. Spatial irreps use
Cotton ordering; double-group bosonic irreps keep those indices and fermionic
irreps follow them. Unlisted irreps have zero occupation. ROHF and CUHF require
the minority-spin count to be no larger than the majority-spin count in each
irrep. GHF has no independent alpha/beta occupation constraints.

Both options may be supplied. A compatible target and configuration are
enforced from the first density construction through the final iteration.
``hf.state_symmetry`` reports the determinant's total irrep. For spin-orbit GHF,
``hf.state_symmetry_weights`` reports full character-projector weights, and the
label is ``None`` if the determinant contains more than one total irrep.

Why energy-based MO grouping failed
-----------------------------------

Let ``S`` be the AO overlap matrix and ``U(g)`` the matrix that transforms AO
coefficient vectors under operation ``g``. The operator's AO matrix is
``S @ U(g)``; ``U(g)`` alone is a permutation/phase transformation, not an AO
overlap integral. For orthonormal MO coefficients ``C``, its representation is

.. math::

   R(g) = C^\dagger S U(g) C.

Symmetry-related orbitals need not have numerically equal orbital energies.
At stretched N2 geometries, mixed core sigma partners can have splittings
larger than the old energy-grouping tolerance. Operation matrices can also
couple nonadjacent MO columns. An energy window therefore need not define an
invariant space, and repeated unconstrained projections can undo symmetry
resolved during earlier passes.

``MOSymmetryDetector`` now builds a connectivity graph from off-diagonal
couplings in every ``R(g)``. Each connected component is treated as a candidate
invariant block, independent of energy ordering. Within that block, it
successively diagonalizes each commuting spatial operation, splitting the
current subspaces by their positive and negative eigenvalues. Later
operations only rotate within previously resolved subspaces.

The projected orbital-energy operator is then diagonalized within each final
irrep subspace. This removes arbitrary rotations between orbitals of the same
symmetry and supplies consistent projected energies. Columns within a coupled
block are sorted by those energies.

Every operation must subsequently have a diagonal representation of unit
magnitude, and each orbital's character vector must match the character table.
Failures raise ``RuntimeError``. There is no fallback that silently assigns the
totally symmetric irrep. The default representation/character tolerance is
``1e-6`` with zero relative tolerance.

Responsibilities and SCF lifecycle
----------------------------------

.. list-table:: Implementation boundaries
   :header-rows: 1
   :widths: 37 63

   * - Component
     - Responsibility
   * - ``symmetry/mo_sym_detect.py``
     - Spatial AO operations, connected MO blocks, and validated spatial labels
   * - ``symmetry/symmetry_basis.py``
     - Orthonormal spatial irrep blocks and adaptation of supplied guesses
   * - ``symmetry/double_group.py``
     - Spin rotations, character projectors, spinor multiplets, determinant weights
   * - ``scf/occupations.py``
     - Option resolution, feasibility checks, and occupation permutations
   * - ``scf/scf_base.py``
     - Shared SCF orchestration, coefficient ordering, and final target verification
   * - ``base_classes/ci_base.py``
     - Compatibility with the current relativistic CI string algebra

The two symmetry-basis implementations expose the same ``eigh(F)`` and
``adapt(C, eps)`` interface. They return energies, coefficients, and irrep
indices together. ``SCFBase`` therefore does not implement separate spatial and
double-group guess-projection paths.

The lifecycle is:

1. Constructor validation checks option types, integer counts, and spin-pair shape.
2. Binding to a system resolves the group's labels and electron counts, checks
   complete occupations, and rejects incompatible targets.
3. At run startup, the orthonormal symmetry basis is built once from the
   canonical AO orthogonalizer. Capacity and target feasibility are checked
   before the first density and J/K build.
4. SAP/core guesses use the same symmetry diagonalization when constraints
   are active. Supplied or perturbed guesses are adapted through the basis
   interface. For a supplied guess without stored energies, the diagonal of
   ``C.conj().T @ H @ C`` supplies an energy proxy for occupation selection.
5. Every subsequent Fock diagonalization operates within symmetry blocks.
   The policy returns column permutations before density construction.
6. Final orbital labels come from diagonalization metadata. The determinant
   symmetry is computed separately and checked against the requested target.

Occupied orbitals precede virtual orbitals, with energy ordering inside each
partition. ROHF additionally separates doubly and singly occupied partitions.
This ordering preserves the existing density builders' occupied-column slices.
Unlike a final rotation used solely to label orbitals, symmetry adaptation
during SCF keeps the orbitals and density consistent.

The raw dataclass options are never replaced by resolved counts or normalized
labels. Rebuilding a method chain can therefore feed them back into the
constructor. Basis blocks and spinor partners are rebuilt at each SCF startup;
rebinding clears determinant labels and projector weights.

Selecting occupations
---------------------

Target-only selection minimizes the sum of current occupied orbital energies,
subject to electron counts and symmetry. This is an occupation rule within SCF,
not a guarantee of the globally lowest HF energy. Distinct stationary
solutions may share the same target irrep.

For a one-dimensional irrep algebra, let ``A(k,h)`` be the smallest energy sum
for ``k`` occupied orbitals with total irrep ``h``. Adding orbital ``p`` of
energy ``epsilon_p`` and irrep ``gamma_p`` gives the recurrence

.. math::

   A_{\mathrm{new}}(k,h)
   = \min\left[
       A_{\mathrm{old}}(k,h),
       A_{\mathrm{old}}(k-1,h\otimes\gamma_p^{-1})+\epsilon_p
     \right].

The initial state is ``A(0, identity) = 0``, with all other entries infinite.
Backtracking returns the selected MO indices. For ordinary spatial irreps,
the products are Cotton-index XOR. For Abelian double groups, the multiplication
table is derived from characters; complex ``+i``/``-i`` irreps cannot use the
spatial XOR rule. This algorithm costs ``O(M N R)`` for ``M`` orbitals,
``N`` occupied orbitals, and ``R`` irreps; its backtracking history has the same
asymptotic storage cost.

UHF computes a spectrum for each spin and minimizes their combined energy
over alpha/beta irreps whose product is the target. It does not choose alpha
occupations independently and then constrain beta.

ROHF and CUHF use a joint dynamic program over doubly occupied counts, singly
occupied counts, and total irrep. Per-irrep energy prefix sums permit evaluation
of each nested core/open occupation. RHF has only a totally symmetric
determinant; a nonidentity target is rejected, while explicit per-irrep pair
counts can still select different closed-shell configurations.

The common ``_OccupationConstraints`` base resolves labels and count arrays and
checks basis capacity. Spatial and double-group policies are separate
subclasses, so shared validation does not impose spatial XOR assumptions on
spinor irreps.

Full double-group representation
--------------------------------

Spinors use the block AO layout ``[alpha AOs, beta AOs]``. A proper pi rotation
about axis ``k`` is lifted to spin space as

.. math::

   D^{1/2}(C_{2k}) = -i\sigma_k,
   \qquad D^{1/2}(E) = D^{1/2}(i) = I_2.

Inversion acts as identity on the spin part of a two-component spinor. A mirror
is inversion times the pi rotation about its normal, so its spin matrix is
also ``-i sigma_normal``. The full AO operator is the Kronecker product of
the spin matrix and the spatial AO transformation.

Each spatial operation has unbarred and barred lifts. The barred operation
differs by a 2pi rotation and has the negative spinor matrix. Bosonic characters
are unchanged by barring; fermionic characters change sign.

C1*, Ci*, C2*, Cs*, and C2h* have one-dimensional irreps. D2* and C2v* have a
two-dimensional fermionic irrep; D2h* has two such irreps, ``e1/2g`` and
``e1/2u``. These noncommuting spin rotations cannot be resolved by the spatial
detector's simultaneous diagonalization procedure.

Instead, ``DoubleGroupBasis`` constructs character projectors in the
orthonormal AO space:

.. math::

   P_\Gamma
   = \frac{d_\Gamma}{|G^*|}
       \sum_{g\in G^*}\chi_\Gamma(g)^*
       X^\dagger S U(g)X,
   \qquad X^\dagger S X=I.

The eigenvectors with projector eigenvalue one span that irrep's isotypic
block, including all equivalent copies. The implementation checks that every
operation is unitary in the retained orbital space, projector eigenvalues
are zero or one, and the blocks form a complete orthonormal basis. Truncating
only one partner of a spinor multiplet therefore raises an error.

A spinor labelled by a two-dimensional irrep belongs to that isotypic block.
Its label does not mean that the individual column is an eigenvector of every
operation. Character products are generally reducible; for example,

.. math::

   E_{1/2,g}\otimes E_{1/2,g}
   = A_g\oplus B_{1g}\oplus B_{2g}\oplus B_{3g}.

Consequently, multiplying two orbital labels cannot identify an even-electron
determinant's total irrep.

Determinant projectors and representability
-------------------------------------------

For occupied orthonormal spinors ``Cocc``, the overlap with the transformed
Slater determinant is

.. math::

   \langle\Phi|\hat U(g)|\Phi\rangle
   = \det(C_{\mathrm{occ}}^\dagger S U(g)C_{\mathrm{occ}}).

The expectation of the total-irrep projector is

.. math::

   w_\Gamma
   = \frac{d_\Gamma}{|G^*|}
       \sum_{g\in G^*}\chi_\Gamma(g)^*
       \det(C_{\mathrm{occ}}^\dagger S U(g)C_{\mathrm{occ}}).

Weights must be nonnegative and sum to one within ``1e-6``. A determinant is
reported as pure if exactly one weight is one within that absolute tolerance.
Odd-electron targets must be fermionic; even-electron targets must be bosonic.

For D2*, C2v*, and D2h*, a determinant in a one-dimensional total irrep must
have an occupied subspace invariant under every group operation. That
subspace consists of complete copies of the two-dimensional spinor irreps.
Each copy has determinant one under all operations, including inversion.
Their product is therefore the totally symmetric irrep. Nonidentity bosonic
targets in these groups require a multideterminant state and are rejected.
This restriction does not apply to the Abelian double groups.

For a totally symmetric target in the non-Abelian double groups, the basis
caches two partner component bases ``X0`` and ``X1``. ``X0`` is the ``-i``
eigenspace of ``C2z``; a second generator maps it to ``X1``. Schur averaging
reduces the Fock matrix to a common multiplicity-space block:

.. math::

   F_{\mathrm{eff}}
   = \tfrac12(X_0^\dagger F X_0 + X_1^\dagger F X_1).

Both partners use the same eigenvectors of this matrix. Their equal energies
and adjacent ordering permit complete-multiplet occupation. This component
trace equals an explicit average over the full double group, and avoids
rebuilding partner bases and all transformed Fock blocks during every iteration.
Explicit counts combined with this target must be even in every spinor irrep.

Counts without a target may describe a mixed even-electron determinant.
The returned projector weights expose that mixture rather than inventing a
single total-irrep label.

Relativistic CI compatibility
-----------------------------

The present CI string machinery implements ordinary spatial XOR products.
Passing full double-group indices to it can incorrectly remove all determinants
for odd electron counts. Spin-orbit ``RelCIBase`` therefore supplies C1 orbital
indices to its workers and rejects a nonzero CI state-symmetry request.
Spin-free CI retains its existing spatial symmetry handling.

This boundary applies to CI and its MCSCF drivers; the new double-group target
belongs to HF and does not constrain a subsequent CI state. Adding full
double-group CI state selection would require a different determinant/state
symmetry treatment, beyond the HF feature.

Reproducing the stretched N2 configuration
------------------------------------------

At 2.5 Angstrom in cc-pVTZ, both integral backends reproduce the following
fixed closed-shell configuration with cc-pVTZ-JKFIT density fitting:

.. code-block:: python

   import forte2

   system = forte2.System(
       xyz="N 0 0 0; N 0 0 2.5",
       basis_set="cc-pvtz",
       auxiliary_basis_set="cc-pvtz-jkfit",
       symmetry=True,
       integral_backend="libint2",  # or "libcint"
   )
   hf = forte2.RHF(
       charge=0,
       irrep_occupations={"ag": 3, "b1u": 2, "b2u": 1, "b3u": 1},
   )(system).run()

The occupied spatial counts are ``(3, 2, 1, 1)`` for the listed irreps, the total
state is ``ag``, and the regression reference energy is
``-108.14839967816553 Eh`` with an absolute tolerance of ``1e-9 Eh``.
Tests evaluate orbitals at real-space points under all spatial operations,
checking every reported label independently of the MO matrix detector.

Correct integrals and correct orbital labels do not imply that unconstrained
SCF will choose this configuration. Different initial guesses can reach
different stationary solutions. ``target_symmetry="ag"`` alone cannot
distinguish them because every closed-shell RHF determinant is totally
symmetric; use explicit occupations when a particular configuration matters.

Verification and maintenance
-----------------------------

The regressions cover:

* Nonadjacent spatial MO mixing with resolvable energy splittings, repeated
  irreps, complex coefficients, and rejection of incomplete or invalid spaces.
* Stretched N2, H2, and Li2, open-shell spatial HF, and the N2 cc-pVTZ
  configuration on both integral backends.
* All five HF occupation conventions, jointly selected UHF spins, nested
  ROHF/CUHF occupations, invalid requests, and early feasibility failures.
* Character orthogonality, group closure, direct products, random complex
  projector spaces, and exhaustive occupation searches for every supported
  Abelian double group.
* Spinor-projector checks in real space, nonzero spin-orbit coupling in water
  and Br2, and distinct complex irreps in C2h trans-diazene.
* Equality between the cached component trace and full group-averaged Fock
  spectra, pure and mixed determinant weights, rebuild/rebind behavior, and
  HF-to-relativistic-CI compatibility.

The relevant files are ``tests/symmetry/test_mo_sym_detect.py``,
``tests/symmetry/test_double_group.py``, ``tests/scf/test_occupations.py``,
and ``tests/ci/test_rel_ci.py``. Run these first, then the non-slow suite:

.. code-block:: console

   conda run -n forte2-dev-1 pytest tests/scf tests/symmetry tests/ci/test_rel_ci.py -m "not slow"
   conda run -n forte2-dev-1 pytest -m "not slow"

Local validation must avoid user-specific Forte2 mods. The validation runner
used for this change temporarily substitutes an empty ``Path.home()`` only
during the initial package import; subsequent tests run with the normal home
directory. No C++ implementation or binding changes are required.

For background on fermionic/bosonic double-group tables and the distinction
between Abelian and non-Abelian binary groups, see the
`DIRAC symmetry documentation <https://www.diracprogram.org/doc/master/manual/groupchain.html>`_.
Forte2's index ordering retains its existing Cotton bosonic indices rather than
using DIRAC's fermion-first ordering.
