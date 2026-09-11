from forte2.lib import ints
from forte2.system import ModelSystem


def validate_df_gradient_system(system, method_name: str) -> None:
    """Validate system-level requirements shared by density-fitted gradients."""
    if isinstance(system, ModelSystem):
        raise NotImplementedError(
            f"{method_name} gradients are not implemented for ModelSystem."
        )
    if system.cholesky_tei:
        raise NotImplementedError(
            f"{method_name} gradients are implemented only for density fitting, "
            "not cholesky_tei."
        )
    if system.auxiliary_basis is None:
        raise NotImplementedError(
            f"{method_name} gradients require an auxiliary basis set for density fitting."
        )

    max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
    if max_l > ints.libint2_max_am:
        raise NotImplementedError(
            f"{method_name} gradients require derivative integrals supported by "
            f"Libint2 (max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
        )
