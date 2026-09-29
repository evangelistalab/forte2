from dataclasses import dataclass, fields

from forte2.helpers import logger
from forte2.orbitals.orbital_overlap import transfer_orbitals
from .method import Method
from .mo import MO


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


@dataclass(frozen=True)
class OrbitalSnapshot:
    """
    SCF orbitals captured from a method chain, for seeding another chain.

    Attributes
    ----------
    system : System
        The system whose AO basis the orbitals are expressed in.
    mos : MO
        A copy of the SCF orbitals.
    scf_class : type
        The class of the SCF method that produced the orbitals.
    """

    system: object
    mos: MO
    scf_class: type


def snapshot_orbitals(method):
    """
    Capture the SCF orbitals of a method chain.

    The snapshot is a copy, so it stays valid after the chain is rebound to
    another geometry.

    Parameters
    ----------
    method : object
        The last stage of the chain.

    Returns
    -------
    OrbitalSnapshot | None
        The snapshot, or None if the chain's SCF has not been run.
    """
    root = list_method_chain(method)[0]
    if not root.executed:
        return None
    return OrbitalSnapshot(
        system=root.system, mos=root.mos.copy(), scf_class=type(root)
    )


def _fresh_copy(obj):
    """
    Reconstruct a method object from its initialization options.
    """
    kwargs = {item.name: getattr(obj, item.name) for item in fields(obj) if item.init}
    return type(obj)(**{name: _fresh_value(v) for name, v in kwargs.items()})


def _fresh_value(value):
    if isinstance(value, Method):
        return _fresh_copy(value)
    if isinstance(value, (list, tuple)) and any(isinstance(v, Method) for v in value):
        return type(value)(_fresh_value(v) for v in value)
    return value


def seed_scf_guess(snapshot, method):
    """
    Seed the SCF of a method chain with orbitals transferred from a snapshot.

    Parameters
    ----------
    snapshot : OrbitalSnapshot
        Orbitals captured with `snapshot_orbitals`.
    method : object
        The last stage of the chain to seed. Its SCF is modified in place.

    Returns
    -------
    bool
        True if a guess was installed, False if the SCF keeps its default guess.
    """
    root = list_method_chain(method)[0]
    if type(root) is not snapshot.scf_class:
        logger.log_warning(
            f"Cannot seed {type(root).__name__} with {snapshot.scf_class.__name__} "
            "orbitals; it will start from its default guess."
        )
        return False

    guess = []
    for C in snapshot.mos.C:
        C_new = transfer_orbitals(C, snapshot.system, root.system)
        if C_new is None:
            return False
        guess.append(C_new)
    root.C = guess
    return True
