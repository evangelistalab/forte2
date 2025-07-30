import numpy as np

from forte2.system import System
from forte2.state import MOSpace
from forte2.jkbuilder import FockBuilder


class Semicanonicalizer:
    def __init__(
        self,
        mo_space: MOSpace,
        g1_sf: np.ndarray,
        C: np.ndarray,
        system: System,
        fock_builder: FockBuilder = None,
        mix_inactive: bool = False,
        mix_active: bool = False,
    ):
        self.mo_space = mo_space
        # factor of 0.5 to use (2J - K) throughout for Fock build
        self.g1_sf = 0.5 * g1_sf
        self.system = system
        self.fock_builder = fock_builder
        self._C = C[:, self.mo_space.orig_to_contig].copy()
        self.mix_inactive = mix_inactive
        self.mix_active = mix_active

        if self.fock_builder is None:
            self.fock_builder = FockBuilder(self.system, use_aux_corr=True)

        self.hcore = self.system.ints_hcore()

    def _build_fock(self):
        # include frozen core in Fock build
        docc = slice(0, self.mo_space.core.stop)
        C_docc = self._C[:, docc]
        J, K = self.fock_builder.build_JK([C_docc])
        fock = self.hcore + 2 * J[0] - K[0]

        C_act = self._C[:, self.mo_space.actv]

        J, K = self.fock_builder.build_JK_generalized(C_act, self.g1_sf)
        fock += 2 * J - K
        fock = np.einsum("pq,pi,qj->ij", fock, self._C.conj(), self._C, optimize=True)
        return fock

    def run(self):
        fock = self._build_fock()
        eps = np.zeros(self.mo_space.nmo)
        U = np.zeros((self.mo_space.nmo, self.mo_space.nmo))

        # core blocks
        if self.mix_inactive and self.mo_space.nfrozen_core + self.mo_space.ncore > 0:
            # semicanonicalize frozen core and core together if mix_inactive
            _core = slice(0, self.mo_space.core.stop)
            e, c = np.linalg.eigh(fock[_core, _core])
            eps[_core] = e
            U[_core, _core] = c
        else:
            if self.mo_space.nfrozen_core > 0:
                e, c = np.linalg.eigh(
                    fock[self.mo_space.frozen_core, self.mo_space.frozen_core]
                )
                eps[self.mo_space.frozen_core] = e
                U[self.mo_space.frozen_core, self.mo_space.frozen_core] = c

            if self.mo_space.ncore > 0:
                e, c = np.linalg.eigh(fock[self.mo_space.core, self.mo_space.core])
                eps[self.mo_space.core] = e
                U[self.mo_space.core, self.mo_space.core] = c

        # active blocks
        if self.mo_space.nactv > 0:
            if self.mix_active:
                # semicanonicalize active orbitals together if mix_active
                e, c = np.linalg.eigh(fock[self.mo_space.actv, self.mo_space.actv])
                eps[self.mo_space.actv] = e
                U[self.mo_space.actv, self.mo_space.actv] = c
            else:
                for igas in range(self.mo_space.ngas):
                    e, c = np.linalg.eigh(
                        fock[self.mo_space.gas[igas], self.mo_space.gas[igas]]
                    )
                    eps[self.mo_space.gas[igas]] = e
                    U[self.mo_space.gas[igas], self.mo_space.gas[igas]] = c

        # virtual blocks
        if (
            self.mix_inactive
            and self.mo_space.nfrozen_virtual + self.mo_space.nvirt > 0
        ):
            # semicanonicalize frozen virtual and virtual together if mix_inactive
            _virt = slice(self.mo_space.virt.start, self.mo_space.frozen_virt.stop)
            e, c = np.linalg.eigh(fock[_virt, _virt])
            eps[_virt] = e
            U[_virt, _virt] = c
        else:
            if self.mo_space.nvirt > 0:
                e, c = np.linalg.eigh(fock[self.mo_space.virt, self.mo_space.virt])
                eps[self.mo_space.virt] = e
                U[self.mo_space.virt, self.mo_space.virt] = c

            if self.mo_space.nfrozen_virtual > 0:
                e, c = np.linalg.eigh(
                    fock[self.mo_space.frozen_virt, self.mo_space.frozen_virt]
                )
                eps[self.mo_space.frozen_virt] = e
                U[self.mo_space.frozen_virt, self.mo_space.frozen_virt] = c

        self.U = U
        self.Uactv = U[self.mo_space.actv, self.mo_space.actv]
        self.C_semican = (self._C @ U)[:, self.mo_space.contig_to_orig]
        self.eps_semican = eps[self.mo_space.contig_to_orig]

        return self
