"""Double groups of D2h and its subgroups, including spin rotations."""

import numpy as np

from .mo_sym_detect import build_ao_symmetry_operations
from .sym_utils import CHARACTER_TABLE, COTTON_LABELS, SYMMETRY_OPS


def spin_rotation(operation):
    """SU(2) lift; inversion acts as identity on two-component spinors."""
    pauli = {
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.diag([1, -1]).astype(complex),
    }
    if operation in ("E", "i"):
        return np.eye(2, dtype=complex)
    axis = {
        "C2x": "x",
        "C2y": "y",
        "C2z": "z",
        "σ_yz": "x",
        "σ_xz": "y",
        "σ_xy": "z",
    }[operation]
    return -1j * pauli[axis]


class DoubleGroup:
    """Character table with unbarred operations followed by their 2π lifts.

    Ordinary irreps retain Cotton indices. Spinorial irreps follow them, with
    1/2 in their labels to distinguish them from ordinary spatial irreps.
    """

    def __init__(self, point_group):
        self.spatial_group = point_group
        self.point_group = point_group + "*"
        self.operations = SYMMETRY_OPS[point_group]
        self.nonabelian = point_group in ("D2", "C2V", "D2H")
        labels = list(COTTON_LABELS[point_group])
        characters = [CHARACTER_TABLE[point_group][label] * 2 for label in labels]
        self.fermion_parities = {}
        parities = (1, -1) if "i" in self.operations else (1,)
        for parity in parities:
            suffix = ("g" if parity == 1 else "u") if len(parities) == 2 else ""
            if self.nonabelian:
                names = ["e1/2" + suffix]
                rows = [
                    [
                        np.trace(spin_rotation(op)) * (parity if op == "i" else 1)
                        for op in self.operations
                    ]
                ]
            elif point_group in ("C1", "CI"):
                names = ["a1/2" + suffix]
                rows = [[parity if op == "i" else 1 for op in self.operations]]
            else:
                names = ["1e1/2" + suffix, "2e1/2" + suffix]
                rows = []
                for phase in (1j, -1j):
                    row = []
                    for op in self.operations:
                        if op == "E":
                            row.append(1)
                        elif op == "i":
                            row.append(parity)
                        elif op == "σ_xy" and point_group == "C2H":
                            row.append(phase * parity)
                        else:
                            row.append(phase)
                    rows.append(row)
            for name, row in zip(names, rows):
                self.fermion_parities[len(labels)] = int(parity == -1)
                labels.append(name)
                characters.append(list(row) + list(-np.array(row)))
        self.labels = {label: index for index, label in enumerate(labels)}
        self.characters = np.array(characters, dtype=complex)
        self.dimensions = self.characters[:, 0].real.astype(int)
        self.order = self.characters.shape[1]
        # Character multiplication gives multiplicities, including reducible
        # products of multidimensional fermionic irreps.
        self.products = np.rint(
            np.einsum(
                "ig,jg,kg->ijk",
                self.characters,
                self.characters,
                self.characters.conj(),
            ).real
            / self.order
        ).astype(int)
        self.abelian_products = (
            np.argmax(self.products, axis=2) if not self.nonabelian else None
        )

    def ao_operations(self, system, info):
        spatial = build_ao_symmetry_operations(system, info, self.spatial_group)
        unbarred = {
            op: np.kron(spin_rotation(op), spatial[op]) for op in self.operations
        }
        return unbarred | {"bar_" + op: -U for op, U in unbarred.items()}

    def determinant_weights(self, Cocc, S, U_ops):
        """Irrep projector expectation values in a normalized Slater determinant."""
        overlaps = np.array(
            [np.linalg.det(Cocc.conj().T @ S @ U @ Cocc) for U in U_ops.values()]
        )
        weights = (
            self.dimensions * (self.characters.conj() @ overlaps).real / self.order
        )
        if np.any(weights < -1e-6) or not np.isclose(
            weights.sum(), 1, atol=1e-6, rtol=0
        ):
            raise RuntimeError("Invalid double-group determinant symmetry weights.")
        return np.clip(weights, 0, 1)

    def determinant_symmetry(self, Cocc, S, U_ops):
        """Return the pure determinant irrep (if any) and all projector weights."""
        weights = self.determinant_weights(Cocc, S, U_ops)
        names = list(self.labels)
        pure = np.flatnonzero(np.isclose(weights, 1, atol=1e-6, rtol=0))
        label = names[int(pure[0])] if len(pure) == 1 else None
        return label, dict(zip(names, weights.tolist()))


class DoubleGroupBasis:
    """Orthonormal isotypic blocks constructed with full character projectors."""

    def __init__(self, group, S, X, U_ops, paired=False):
        self.group = group
        self.point_group = group.point_group
        self.S = S
        self.U_ops = U_ops
        self.paired = paired
        self.blocks = {}
        representations = np.array([X.conj().T @ S @ U @ X for U in U_ops.values()])
        identity = np.eye(X.shape[1])
        if any(
            not np.allclose(R.conj().T @ R, identity, atol=1e-6, rtol=0)
            for R in representations
        ):
            raise RuntimeError("AO space is not closed under double-group operations.")
        for irrep in group.fermion_parities:
            projector = (
                group.dimensions[irrep]
                * np.einsum(
                    "g,gij->ij", group.characters[irrep].conj(), representations
                )
                / group.order
            )
            eigenvalues, vectors = np.linalg.eigh(projector)
            if not np.allclose(eigenvalues, np.round(eigenvalues), atol=1e-6, rtol=0):
                raise RuntimeError(
                    "AO space is not closed under double-group operations."
                )
            self.blocks[irrep] = X @ vectors[:, eigenvalues > 0.5]
        self.C = np.column_stack(list(self.blocks.values()))
        self.irreps = np.concatenate(
            [
                np.full(block.shape[1], irrep, dtype=int)
                for irrep, block in self.blocks.items()
            ]
        )
        if self.C.shape != X.shape or not np.allclose(
            self.C.conj().T @ S @ self.C, identity, atol=1e-6, rtol=0
        ):
            raise RuntimeError("Double-group projectors do not span the orbital space.")
        self._multiplets = {}
        if paired:
            self._build_multiplets()

    def _build_multiplets(self):
        """Cache the two component bases of each equivalent spinor multiplet."""
        partner_op = "C2y" if "C2y" in self.U_ops else "σ_xz"
        for irrep, X in self.blocks.items():
            rotation = X.conj().T @ self.S @ self.U_ops["C2z"] @ X
            phase, vectors = np.linalg.eigh(1j * rotation)
            first = X @ vectors[:, phase > 0.5]
            second = self.U_ops[partner_op] @ first
            self._multiplets[irrep] = (first, second)

    @classmethod
    def build(cls, system, info, S, X, group, paired=False):
        return cls(group, S, X, group.ao_operations(system, info), paired)

    def eigh(self, F):
        energies, coefficients, irreps = [], [], []
        for irrep, X in self.blocks.items():
            if not X.shape[1]:
                continue
            if self.paired:
                # A one-dimensional many-electron irrep requires an invariant
                # occupied subspace. Fill complete 2D spinor multiplets. Schur's
                # lemma makes their determinant the totally symmetric irrep.
                first, second = self._multiplets[irrep]
                # The spinor-component trace equals the full group average
                # in the multiplicity space, without rebuilding every U @ X.
                Fhalf = 0.5 * (
                    first.conj().T @ F @ first + second.conj().T @ F @ second
                )
                e, c = np.linalg.eigh(Fhalf)
                C = np.empty((X.shape[0], 2 * len(e)), dtype=complex)
                C[:, ::2], C[:, 1::2] = first @ c, second @ c
                e = np.repeat(e, 2)
            else:
                e, c = np.linalg.eigh(X.conj().T @ F @ X)
                C = X @ c
            energies.extend(e)
            coefficients.append(C)
            irreps.extend([irrep] * len(e))
        energies = np.array(energies)
        order = np.argsort(energies, kind="stable")
        return (
            energies[order],
            np.column_stack(coefficients)[:, order],
            np.array(irreps)[order],
        )

    def adapt(self, C, eps):
        """Project a supplied guess using its orbital-energy operator."""
        SC = self.S @ C
        return self.eigh((SC * eps) @ SC.conj().T)
