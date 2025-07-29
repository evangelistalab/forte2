import numpy as np
import scipy as sp

from forte2.system import System
from forte2.state import MOSpace
from forte2.jkbuilder import FockBuilder


class Semicanonicalizer:
    def __init__(self, mo_space: MOSpace, g1_sf, C, system: System, fock_builder=None):
        self.mo_space = mo_space
        # factor of 0.5 to use (2J - K) throughout for Fock build
        # (i.e., gamma_core = CC^+ instead of 2CC^+)
        self.g1_sf = 0.5 * g1_sf
        self.system = system
        self.fock_builder = fock_builder
        self._C = C[:, self.mo_space.orig_to_contig].copy()

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
        try:
            L = np.linalg.cholesky(self.g1_sf, upper=False)
            Cp = C_act @ L
        except np.linalg.LinAlgError:
            n, L = np.linalg.eigh(self.g1_sf)
            assert np.all(n > -1.0e-11), "g1_sf must be positive semi-definite"
            n = np.maximum(n, 0)
            Cp = C_act @ L @ np.diag(np.sqrt(n))
        J, K = self.fock_builder.build_JK([Cp])
        fock += 2 * J[0] - K[0]
        fock = np.einsum("pq,pi,qj->ij", fock, self._C.conj(), self._C, optimize=True)
        return fock

    def run(self):
        fock = self._build_fock()
        eps = np.zeros(self.mo_space.nmo)
        U = np.zeros((self.mo_space.nmo, self.mo_space.nmo))

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

        if self.mo_space.nactv > 0:
            for igas in range(self.mo_space.ngas):
                e, c = np.linalg.eigh(
                    fock[self.mo_space.gas[igas], self.mo_space.gas[igas]]
                )
                eps[self.mo_space.gas[igas]] = e
                U[self.mo_space.gas[igas], self.mo_space.gas[igas]] = c

        if self.mo_space.nvirt > 0:
            e, c = np.linalg.eigh(fock[self.mo_space.virt, self.mo_space.virt])
            eps[self.mo_space.virt] = e
            U[self.mo_space.virt, self.mo_space.virt] = c

        if self.mo_space.nfrozen_virtual > 0:
            e, c = np.linalg.eigh(
                fock[self.mo_space.frozen_virtual, self.mo_space.frozen_virtual]
            )
            eps[self.mo_space.frozen_virtual] = e
            U[self.mo_space.frozen_virtual, self.mo_space.frozen_virtual] = c

        self.U = U
        self.Uactv = U[self.mo_space.actv, self.mo_space.actv]
        C_semican = (U.T.conj() @ self._C @ U)[:, self.mo_space.contig_to_orig]
        eps_semican = eps[self.mo_space.contig_to_orig]
        return eps_semican, C_semican
