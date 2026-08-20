from dataclasses import fields

import numpy as np
import scipy as sp

import forte2.integrals as integrals
from .method import Method


def list_method_chain(method):
    """
    Collect the stages of a method chain, from the root to `method`.

    The root is the stage bound directly to a ``System`` (an SCF object); every
    other stage is bound to its predecessor through ``parent_method``.

    Parameters
    ----------
    method : object
        The last stage of the chain.

    Returns
    -------
    list
        The chain stages ordered root first.

    Raises
    ------
    ValueError
        If the chain contains a cycle.
    """
    stages = []
    seen = set()
    stage = method
    while stage is not None:
        if id(stage) in seen:
            raise ValueError("Method chain contains a cycle and cannot be rebuilt.")
        seen.add(id(stage))
        stages.append(stage)
        stage = getattr(stage, "parent_method", None)
    return stages[::-1]


def rebuild_method_chain(method, new_system):
    """
    Rebuild an entire method chain against `new_system`.
    The input method chain is left untouched.

    Parameters
    ----------
    method : object
        The last stage (leaf) of the chain to reproduce.
    new_system : System
        The system to bind the rebuilt chain to.

    Returns
    -------
    object
        The last stage of the rebuilt chain, bound to `new_system` but not run.
    """
    stages = list_method_chain(method)
    rebuilt = _fresh_copy(stages[0])(new_system)
    for stage in stages[1:]:
        rebuilt = _fresh_copy(stage)(rebuilt)
    return rebuilt


def reset_method_chain(method):
    """
    Invalidate every stage of a method chain in place, root first.

    Parameters
    ----------
    method : object
        The last stage (leaf) of the chain to invalidate.

    Returns
    -------
    object
        `method`, with every stage's run() results invalidated.
    """
    for stage in list_method_chain(method):
        stage.reset()
    return method


def rebind_method_chain(method, new_system):
    """
    Reattach an existing method chain to `new_system` in place.

    Unlike `rebuild_method_chain`, this does not construct new objects: every
    stage is reset (see `reset_method_chain`) and then re-`__call__`ed onto its
    (already rebound) predecessor, so the same chain can be reused across many
    geometries without reallocating it.

    Parameters
    ----------
    method : object
        The last stage (leaf) of the chain to rebind.
    new_system : System
        The system to bind the root of the chain to.

    Returns
    -------
    object
        `method`, now bound to `new_system` but not run.
    """
    stages = list_method_chain(method)
    reset_method_chain(method)
    upstream = stages[0](new_system)
    for stage in stages[1:]:
        upstream = stage(upstream)
    return method


class _OrbitalSnapshot:
    """A frozen (system, MO coefficients) pair with the duck type that
    `seed_chain_orbitals`/`project_occupied_orbitals` expect from a source
    method: a `.system` and a `.mos.C`."""

    __slots__ = ("system", "mos")

    def __init__(self, system, C):
        self.system = system
        self.mos = _MOSnapshot(C)


class _MOSnapshot:
    __slots__ = ("C",)

    def __init__(self, C):
        self.C = C


def snapshot_orbitals(method):
    """
    Capture a method's current system and MO coefficients, decoupled from any
    later mutation of `method` itself.

    Needed to seed a rebound chain from what a source method looked like
    *before* it gets reset and rebound to a new geometry, which matters once
    the source and target are the same reused object.

    Parameters
    ----------
    method : object
        A method exposing `.system` and `.mos.C`.

    Returns
    -------
    object | None
        A snapshot usable anywhere a source method is expected, or None if
        `method` has no orbitals yet.
    """
    mos = getattr(method, "mos", None)
    if mos is None or mos.C is None:
        return None
    return _OrbitalSnapshot(method.system, [C.copy() for C in mos.C])


def _fresh_copy(obj):
    """
    Reconstruct a method object from its initialization options.

    Some methods overwrite an initialization field with a value derived from the
    others -- ``ActiveSpaceSolver.mo_space`` is built from the ``*_orbitals``
    lists, ``RelActiveSpaceSolver.states`` from ``nel`` -- and replaying both the
    derived value and the arguments it came from is rejected as ambiguous. Such a
    class declares a ``_rebuild_kwargs()`` method returning the options the user
    actually supplied; everything else is reconstructed from its current field
    values.
    """
    if hasattr(obj, "_rebuild_kwargs"):
        kwargs = obj._rebuild_kwargs()
    else:
        kwargs = {
            item.name: getattr(obj, item.name) for item in fields(obj) if item.init
        }
    return type(obj)(**{name: _fresh_value(v) for name, v in kwargs.items()})


def _fresh_value(value):
    if isinstance(value, Method):
        return _fresh_copy(value)
    if isinstance(value, (list, tuple)) and any(isinstance(v, Method) for v in value):
        return type(value)(_fresh_value(v) for v in value)
    return value


def seed_chain_orbitals(source_method, method):
    """
    Seed a rebuilt chain with orbitals projected from an already-converged one.

    The guess is installed on the SCF root of the chain, which is where it is
    consumed: downstream stages take their starting orbitals from their parent,
    so seeding the last stage would have no effect. It is written to the root's
    raw ``C`` list, which is what the SCF loop reads as its initial guess; the
    ``mos`` wrapper is built from it once the root has run.

    Parameters
    ----------
    source_method : object
        A converged method supplying the orbitals to project.
    method : object
        The last stage of the chain to seed. Its root is modified in place.

    Returns
    -------
    bool
        True if a guess was installed, False if the projection does not apply.
    """
    root = list_method_chain(method)[0]
    projected = project_occupied_orbitals(source_method, root)
    if projected is None:
        return False
    root.C = projected
    return True


def project_occupied_orbitals(source_method, method):
    """
    Project occupied orbitals from ``source_method`` into ``method.system``.

    The projection uses the cross-overlap between the new and old AO bases:

    ``Q_occ = X_new^T S(new, old) C_occ_old``,

    where ``X_new`` is the canonical orthogonalizer for the new AO basis. The
    projected occupied subspace is orthonormalized, then completed with an
    orthonormal virtual complement so the SCF object receives a full MO guess.

    Parameters
    ----------
    source_method : object
        A converged method whose orbitals are used as the source.
    method : object
        The method whose system defines the target AO basis.

    Returns
    -------
    list[NDArray] | None
        The projected MO coefficients, or None if the projection does not apply
        (two-component systems, mismatched basis sets, unsupported references).
    """
    if not _can_project_orbitals(source_method, method):
        return None

    source_C = source_method.mos.C
    occupied_counts = _occupied_counts(method)
    if occupied_counts is None or len(occupied_counts) != len(source_C):
        return None

    projected = []
    for C_old, nocc in zip(source_C, occupied_counts):
        C_new = _project_mo_coefficients(
            source_method.system, method.system, C_old, nocc
        )
        if C_new is None:
            return None
        projected.append(C_new)

    return projected


def _can_project_orbitals(source_method, method):
    if getattr(source_method, "mos", None) is None:
        return False
    if getattr(source_method.system, "two_component", False):
        return False
    if getattr(method.system, "two_component", False):
        return False
    if source_method.system.basis_set != method.system.basis_set:
        return False
    if len(source_method.mos.C) not in [1, 2]:
        return False
    return True


def _occupied_counts(method):
    method_name = (
        method._scf_type() if hasattr(method, "_scf_type") else type(method).__name__
    )
    if method_name == "GHF":
        return None
    if not hasattr(method, "na") or not hasattr(method, "nb"):
        return None
    if method_name in ["UHF", "CUHF"]:
        return [method.na, method.nb]
    return [max(method.na, method.nb)]


def _project_mo_coefficients(old_system, new_system, C_old, nocc):
    X_new = new_system.get_Xorth()
    if nocc == 0:
        return X_new.copy()
    if nocc > C_old.shape[1] or nocc > X_new.shape[1]:
        return None

    S_cross = integrals.overlap(new_system, new_system.basis, old_system.basis)
    Q_occ_raw = X_new.T.conj() @ S_cross @ C_old[:, :nocc]
    singular_values = np.linalg.svd(Q_occ_raw, compute_uv=False)
    if len(singular_values) < nocc or singular_values[-1] < 1.0e-8:
        return None

    Q_occ, _ = np.linalg.qr(Q_occ_raw, mode="reduced")
    Q_occ = Q_occ[:, :nocc]

    nvirt = X_new.shape[1] - nocc
    if nvirt > 0:
        Q_virt = sp.linalg.null_space(Q_occ.T.conj())
        if Q_virt.shape[1] < nvirt:
            return None
        Q = np.hstack((Q_occ, Q_virt[:, :nvirt]))
    else:
        Q = Q_occ

    return X_new @ Q
