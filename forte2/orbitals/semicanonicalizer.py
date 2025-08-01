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

        def _eigh(sl):
            return np.linalg.eigh(fock[sl, sl])

        slice_list = []
        if self.mix_inactive:
            slice_list.append(self.mo_space.docc)
        else:
            slice_list.append(self.mo_space.frozen_core)
            slice_list.append(self.mo_space.core)
        if self.mix_active:
            slice_list.append(self.mo_space.actv)
        else:
            slice_list.extend(self.mo_space.gas)
        if self.mix_inactive:
            slice_list.append(self.mo_space.uocc)
        else:
            slice_list.append(self.mo_space.virt)
            slice_list.append(self.mo_space.frozen_virt)

        for sl in slice_list:
            if sl.stop - sl.start > 0:  # Skip empty slices
                e, c = _eigh(sl)
                eps[sl] = e
                U[sl, sl] = c

        self.U = U
        self.Uactv = U[self.mo_space.actv, self.mo_space.actv]
        self.C_semican = (self._C @ U)[:, self.mo_space.contig_to_orig]
        self.eps_semican = eps[self.mo_space.contig_to_orig]

        return self
