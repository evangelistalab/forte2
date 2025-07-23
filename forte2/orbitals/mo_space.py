from dataclasses import dataclass, field


@dataclass
class MOSpace:
    active_spaces: list[int] | list[list[int]]
    core_orbitals: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.active_spaces, list), "Orbitals must be a list."
        if all(isinstance(x, int) for x in self.active_spaces):
            self.ngas = 1
            self.active_spaces = [self.active_spaces]
        elif all(isinstance(x, list) for x in self.active_spaces):
            for sublist in self.active_spaces:
                assert all(
                    isinstance(x, int) for x in sublist
                ), "All elements in the sublists must be integers."
            self.ngas = len(self.active_spaces)

        self.nactv = sum(len(sublist) for sublist in self.active_spaces)
        self.ncore = len(self.core_orbitals)
        # store a flattened list of all active orbitals
        self.active_orbitals = [
            orb for sublist in self.active_spaces for orb in sublist
        ]
        assert (
            len(set(self.active_orbitals)) == self.nactv
        ), "Active orbitals must be unique."
        assert len(set(self.core_orbitals)) == self.ncore, "Core orbitals must be unique."
