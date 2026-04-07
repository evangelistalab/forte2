from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import sys
import time
import resource

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf import UHF
from forte2.helpers import logger


@dataclass
class MP2Base(SystemMixin, MOsMixin, ABC):
    """Base class for density-fitted MP2 methods. Not meant to be used directly.

    Attributes
    ----------
    parent_method : RHF, ROHF, or UHF
        Reference wavefunction object providing orbitals, orbital energies,
        occupation numbers, and Fock builder.
    C : ndarray
        Molecular orbital coefficient matrix in AO basis.
    eps : ndarray
        Orbital energies in the working (possibly semicanonical) basis.
    nocc : int
        Number of correlated occupied orbitals.
    nvir : int
        Number of correlated virtual orbitals.
    B_iaQ : ndarray
        Density-fitted three-index integrals (ia|Q).
    t2 : ndarray
        MP2 double-excitation amplitudes.
    E_corr : float
        MP2 correlation energy.
    E_total : float
        Total energy (E_reference + E_corr).
    executed : bool
        Whether the MP2 calculation has been executed.

    Note
    -----
    This base class assumes spin-restricted tensors:
        B: (nocc, nvir, naux)
        t2: (nocc, nocc, nvir, nvir)

    UHF overrides:
        - _build_df_iaQ
        - _build_t2_all
        - RDM builders

    and does NOT use base-class tensor conventions.
    """

    store_t2: bool = False

    executed: bool = field(default=False, init=False)

    @abstractmethod
    def __call__(self, parent_method): ...

    @abstractmethod
    def make_1rdm(self): ...

    @abstractmethod
    def make_2rdm(self, gamma1=None): ...

    @abstractmethod
    def make_2cumulant(self, gamma1=None, gamma2=None): ...

    @abstractmethod
    def make_cumulants(self, gamma1=None, gamma2=None): ...

    @abstractmethod
    def energy_given_rdms(self, Ecore, H, V, gamma1, gamma2): ...

    def run(self):
        t0 = time.monotonic()
        mem0 = self._memory_snapshot()

        self._startup()

        self.B_iaQ = self._build_df_iaQ()

        self.t2, self.t2_as, self.E_corr = self._build_t2_all(self.B_iaQ)

        self.E_total = self.parent_method.E + self.E_corr

        self.executed = True
        self._log_completion(time.monotonic() - t0, self._t2_norm(), mem0)
        return self.E_total

    def _memory_snapshot(self):
        return {
            "peak_rss_mb": self._peak_rss_mb(),
            "current_rss_mb": self._current_rss_mb(),
        }

    def _peak_rss_mb(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return peak / 1024**2
        return peak / 1024.0

    def _current_rss_mb(self):
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

    def _log_memory(self, start_mem):
        end_peak = self._peak_rss_mb()
        peak_delta = max(0.0, end_peak - start_mem["peak_rss_mb"])
        logger.log_info1(f"Peak RSS = {end_peak:.1f} MB (delta {peak_delta:.1f} MB)")

        end_current = self._current_rss_mb()
        start_current = start_mem["current_rss_mb"]
        if end_current is not None:
            if start_current is not None:
                current_delta = end_current - start_current
                logger.log_info1(
                    f"Current RSS = {end_current:.1f} MB (delta {current_delta:.1f} MB)"
                )
            else:
                logger.log_info1(f"Current RSS = {end_current:.1f} MB")

    def _log_completion(self, elapsed: float, t2_norm: float | None, start_mem):
        logger.log_info1(f"{self._reference_label()}-MP2 calculation completed.")
        logger.log_info1(f"E(corr) = {self.E_corr:.13f} Eh")
        logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
        if t2_norm is None:
            logger.log_info1("||t2|| = not stored")
        else:
            logger.log_info1(f"||t2|| = {t2_norm}")
        self._log_memory(start_mem)
        logger.log_info1(f"Time = {elapsed:.3f} s")

    def _reference_label(self) -> str:
        return "MP2"

    def _t2_norm(self):
        if getattr(self, "t2", None) is None:
            return None
        return np.linalg.norm(self.t2)


    def _safe_divide(self, num, denom, out=None, tiny=1e-12, label="MP2 denom"):
        mask = np.abs(denom) < tiny
        n_bad = np.count_nonzero(mask)
        if n_bad:
            logger.log_warning(f"{label}: {n_bad} / {denom.size} elements < {tiny:g}")
        if out is None:
            return np.divide(num, np.where(mask, np.inf, denom))
        else:
            return np.divide(num, np.where(mask, np.inf, denom), out=out)

    @abstractmethod
    def _build_df_iaQ(self): ...

    @abstractmethod
    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        self.fock_builder = self.system.fock_builder

    @abstractmethod
    def _build_t2_all(self, B): ...
