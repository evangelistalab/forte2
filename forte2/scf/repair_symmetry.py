from dataclasses import dataclass, field

from forte2.base_classes.mixins import MOsMixin, SystemMixin
from forte2.helpers import logger


@dataclass
class RepairSymmetry(MOsMixin, SystemMixin):
    r"""
    Repairs a symmetry-broken SCF reference for post-HF methods.

    This class is intended to be inserted between a converged SCF object and
    methods such as CI, CASSCF/GASSCF, or DSRG:

    >>> ref = RepairSymmetry()(rhf)
    >>> mc = MCOptimizer(...)(ref)

    The upstream SCF object is left untouched. The repaired orbitals are
    non-canonical and should only be used to seed downstream post-HF methods.
    """

    executed: bool = field(default=False, init=False)
    success: bool = field(default=False, init=False)
    repaired: bool = field(default=False, init=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        self.reference_method = getattr(
            parent_method, "reference_method", parent_method
        )
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        required_attrs = ["F", "eps", "mosym", "_diagonalize_fock"]
        for attr in required_attrs:
            assert hasattr(
                self.parent_method, attr
            ), f"Parent method must provide '{attr}' to repair symmetry."

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        self.reference_method = getattr(
            self.parent_method, "reference_method", self.parent_method
        )

        self.eps = [arr.copy() for arr in self.parent_method.eps]
        self.F = [arr.copy() for arr in self.parent_method.F]

        for attr in (
            "E",
            "charge",
            "nel",
            "na",
            "nb",
            "ms",
            "basis_info",
            "nbf",
            "nmo",
            "naux",
            "method",
            "S2",
        ):
            if hasattr(self.parent_method, attr):
                setattr(self, attr, getattr(self.parent_method, attr))

        self.success = self._assign_orbital_symmetries(repair=False)
        if self.success:
            self.executed = True
            return self

        self.F = self.mosym.symmetrize_operator(self.F)
        self.eps, self.C = self.reference_method._diagonalize_fock(self.F)
        self.success = self._assign_orbital_symmetries(repair=True)
        self.repaired = self.success

        if self.success:
            logger.log_info1(
                "Symmetry repaired. Orbital coefficients are non-canonical and should only be used as input to downstream post-HF methods (e.g. CI, CASSCF, DSRG)."
            )

        self.executed = True
        return self

    def _assign_orbital_symmetries(self, repair=False):
        if len(self.C) == 1:
            labels, indices, C_sym, _, success = self.mosym.run(
                self.C[0], self.eps[0], repair=repair
            )
            self.C[0] = C_sym
            self.irrep_labels = labels
            self.irrep_indices = indices
            return success

        if len(self.C) == 2:
            a_labels, a_indices, C_a, _, a_success = self.mosym.run(
                self.C[0], self.eps[0], repair=repair
            )
            b_labels, b_indices, C_b, _, b_success = self.mosym.run(
                self.C[1], self.eps[1], repair=repair
            )
            self.C[0] = C_a
            self.C[1] = C_b
            self.irrep_labels = [a_labels, b_labels]
            self.irrep_indices = [a_indices, b_indices]
            return a_success and b_success

        raise RuntimeError(
            f"RepairSymmetry only supports one- or two-block orbital references, got {len(self.C)}."
        )
