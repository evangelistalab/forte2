import numpy as np
from numpy.typing import NDArray
import forte2
from forte2.base_classes.mixins import MOsMixin, SystemMixin


def _antisymmetrize_pairwise(t4: np.ndarray) -> np.ndarray:
    r"""
    Antisymmetrize a rank-4 tensor in the first and second index pairs.

    This projects a 4-index tensor :math:`T_{pqrs}` onto the subspace that is
    antisymmetric under exchange of the first pair :math:`(p\leftrightarrow q)`
    and antisymmetric under exchange of the second pair :math:`(r\leftrightarrow s)`:

    .. math::

        T^{(A)}_{pqrs} = \frac{1}{4}(1 - P_{pq})(1 - P_{rs})\, T_{pqrs}

    where :math:`P_{pq}` swaps indices :math:`p` and :math:`q`, and :math:`P_{rs}` swaps
    :math:`r` and :math:`s`.

    The factor of :math:`1/4` ensures that if the input tensor is already antisymmetric
    in both pairs, it is left unchanged.

    Parameters
    ----------
    t4
        A rank-4 tensor with shape ``(n, n, n, n)``.

    Returns
    -------
    NDArray
        The pairwise-antisymmetrized tensor, same shape as the input.
    """
    Tp = np.swapaxes(t4, 0, 1)  # P_pq
    Tr = np.swapaxes(t4, 2, 3)  # P_rs
    Tpr = np.swapaxes(Tp, 2, 3)  # P_pq P_rs
    return 0.25 * (t4 - Tp - Tr + Tpr)


class MP2CumulantBuilder(SystemMixin, MOsMixin):
    r"""
    Build approximate MP2 spin-resolved two-body cumulants for mutual-correlation analysis.

    This builder is intended as a *low-cost heuristic* to seed orbital/active-space choices
    (e.g., as an alternative to AVAS). The objects produced here are **not** guaranteed to be
    fully N-representable cumulants, and exact permutation/trace sum rules may be violated
    at a small level.

    Reference and orbital space
    ---------------------------
    - Reference: **RHF** in canonical spatial orbitals.
    - Orbitals are partitioned into occupied (``occ``) and virtual (``virt``).
      There is no explicit “active” space at the RHF level; this implementation builds
      tensors over the full MO list by default.

    Spin-block conventions (spatial-orbital form)
    ---------------------------------------------
    We construct three spin blocks of the two-body cumulant:

    - :math:`\lambda^{\alpha\alpha}_{pqrs}`  (``λaa``)
    - :math:`\lambda^{\alpha\beta}_{pqrs}`  (``λab``)
    - :math:`\lambda^{\beta\beta}_{pqrs}`   (``λbb``)

    where indices :math:`p,q,r,s` label **spatial** MOs and the spin label is carried
    by the block (aa/ab/bb). Same-spin blocks are antisymmetric under :math:`p\leftrightarrow q`
    and :math:`r\leftrightarrow s`; the opposite-spin block is not pairwise antisymmetric.

    MP2 amplitudes used
    -------------------
    For canonical RHF, the (spatial) MP2 double-excitation amplitudes are formed as

    .. math::

        t^{ab}_{ij} = \frac{(ij|ab)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
        \qquad\text{(opposite spin / OS)}

        \\
        t^{ab}_{ij} = \frac{(ij||ab)}{\varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b}
        \qquad\text{(same spin / SS)}

    where
    :math:`(ij|ab)` is the Coulomb integral and :math:`(ij||ab) = (ij|ab) - (ij|ba)` is the
    antisymmetrized integral.

    Approximate 1-RDM
    -----------------
    We also build an approximate spin-dependent 1-RDM :math:`\gamma^\alpha, \gamma^\beta`
    (used by downstream analysis code) using a simple doubles-only :math:`t^2` correction:

    .. math::

        \Delta\gamma_{ij} = -\frac{1}{2}\sum_{kab} t^{ab}_{ik} t^{ab}_{jk},\qquad
        \Delta\gamma_{ab} = +\frac{1}{2}\sum_{ijc} t^{ac}_{ij} t^{bc}_{ij}

    This is *not* the full MP2 Lagrangian density (no :math:`\Lambda` multipliers), so
    the particle-number trace may deviate slightly from the exact value.

    Outputs
    -------
    After calling :meth:`run`, the object has:
    - ``λaa, λab, λbb``: rank-4 cumulant blocks over MOs, shape ``(nmo,nmo,nmo,nmo)``
    - ``γa, γb``: spin-dependent 1-RDMs, shape ``(nmo,nmo)``
    - ``Γ1 = γa + γb``: spin-summed 1-RDM, shape ``(nmo,nmo)``
    """

    def __call__(self, parent_method):
        """
        Initialize the MP2CumulantBuilder with a parent RHF method.
        """
        assert isinstance(
            parent_method, forte2.scf.RHF
        ), f"parent method must be an RHF object, got {type(parent_method)}"

        self.parent_method = parent_method

        return self

    def run(self):
        """
        Execute RHF if needed. Run the MP2 cumulant construction.
        """
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        self.mo_space = self.parent_method.mo_space
        self.nocc = self.parent_method.na
        self.nmo = self.mo_space.nmo

        self.C = self.parent_method.C[0]
        self.eps = self.parent_method.eps[0]

        self.jkbuilder = self.system.fock_builder

        self.orbital_indices = list(range(self.nmo))

        self.λaa, self.λab, self.λbb, self.γa, self.γb, self.Γ1 = (
            self.build_mp2_objects()
        )

        self.executed = True
        return self

    def build_mp2_objects(self) -> tuple[NDArray, NDArray, NDArray]:
        r"""
        Build MP2 cumulant blocks (λaa, λab, λbb) and approximate 1-RDMs (γa, γb, Γ1).

        Indexing and shapes
        -------------------
        Let ``nmo`` be the number of spatial MOs. Then:

        - ``γa, γb, Γ1`` have shape ``(nmo, nmo)``.
        - ``λaa, λab, λbb`` have shape ``(nmo, nmo, nmo, nmo)`` and use the
          convention ``λ[p,q,r,s]`` corresponding to operator ordering
          :math:`a_p^\dagger a_q^\dagger a_s a_r` in the underlying 2-RDM conventions
          used by Forte2 (ensure downstream code uses the same).

        Construction outline
        --------------------
        1) Form occupied/virtual MO lists from RHF occupancy.
        2) Build MO-basis two-electron integral blocks:

           - ``Vc[i,j,a,b] = (ij|ab)``
           - ``Va[i,j,a,b] = (ij||ab)``

        3) Form MP2 denominators :math:`\Delta_{ij}^{ab}` and amplitudes:

           - ``t_os = Vc / denom``  (OS-like; uses Coulomb)
           - ``t_ss = Va / denom``  (SS-like; uses antisymmetrized integrals)

        4) Build an approximate correlated 1-RDM correction from :math:`t^2`
           contractions (doubles-only, no multipliers).

        5) Populate a subset of cumulant tensor blocks using linear and quadratic
           terms in :math:`t`. Then pairwise antisymmetrize same-spin blocks.

        Notes
        -----
        This routine allocates full ``(nmo^4)`` cumulant tensors; for large ``nmo``,
        consider restricting ``orbital_indices`` or switching to a blocked/sparse strategy.

        Returns
        -------
        (λaa, λab, λbb, γa, γb, Γ1)
            The cumulant blocks and 1-RDMs.
        """
        occ = list(range(self.nocc))
        virt = list(range(self.nocc, self.nmo))

        nmo = self.nmo

        nocc = len(occ)
        nvir = len(virt)

        Co = self.C[:, occ]
        Cv = self.C[:, virt]

        # Two-electron MO integrals:
        # V_coul[i,j,a,b] = (ij|ab)
        # V_as  [i,j,a,b] = (ij||ab) = (ij|ab) - (ij|ba)
        V_coul = self.jkbuilder.two_electron_integrals_gen_block(Co, Co, Cv, Cv)
        V_as = self.jkbuilder.two_electron_integrals_gen_block(Co, Co, Cv, Cv, True)

        # The generator returns arrays already in (i,j,a,b) ordering for the provided blocks.
        # Indices here are local to (occ,virt) blocks, hence range(nocc)/range(nvir).
        occ_idx = range(nocc)
        virt_idx = range(nvir)
        Vc = V_coul[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]
        Va = V_as[np.ix_(occ_idx, occ_idx, virt_idx, virt_idx)]

        # MP2 denominators Δ_{ij}^{ab}
        eps_o = self.eps[occ]  # (nocc,)
        eps_v = self.eps[virt]  # (nvir,)
        denom = (
            eps_o[:, None, None, None]
            + eps_o[None, :, None, None]
            - eps_v[None, None, :, None]
            - eps_v[None, None, None, :]
        )

        # Avoid division by small denominators
        tiny = 1e-12
        denom = np.where(np.abs(denom) < tiny, np.inf, denom)

        # Opposite-spin-like and same-spin-like amplitude tensors
        t_os = Vc / denom
        t_ss = Va / denom

        # Approximate spin-dependent 1-RDMs: γa, γb (RHF => γa == γb)
        γa = np.zeros((nmo, nmo))
        γb = np.zeros_like(γa)
        γa[np.ix_(occ, occ)] = np.eye(nocc)
        γb[np.ix_(occ, occ)] = np.eye(nocc)

        # Δγ_oo = -1/2 Σ_{kab} t_{ik}^{ab} t_{jk}^{ab}
        dγoo = -0.5 * (
            np.einsum("ikab,jkab->ij", t_ss, t_ss, optimize=True)
            + np.einsum("ikab,jkab->ij", t_os, t_os, optimize=True)
        )
        # Δγ_vv = +1/2 Σ_{ijc} t_{ij}^{ac} t_{ij}^{bc}
        dγvv = 0.5 * (
            np.einsum("ijac,ijbc->ab", t_ss, t_ss, optimize=True)
            + np.einsum("ijac,ijbc->ab", t_os, t_os, optimize=True)
        )

        # Enforce Hermiticity on the corrections
        dγoo = 0.5 * (dγoo + dγoo.T)
        dγvv = 0.5 * (dγvv + dγvv.T)

        γa[np.ix_(occ, occ)] += dγoo
        γa[np.ix_(virt, virt)] += dγvv
        γb[:] = γa[:]

        Γ1 = γa + γb

        # Spin-dependent 2-body cumulant blocks: λaa, λab, λbb
        λaa = np.zeros((nmo, nmo, nmo, nmo))
        λab = np.zeros_like(λaa)
        λbb = np.zeros_like(λaa)

        # 1. linear ijab blocks
        #   λ^{aa}_{ijab} ~ t_ss
        #   λ^{ab}_{ijab} ~ t_os
        #   λ^{bb}_{ijab} ~ t_ss
        λaa[np.ix_(occ, occ, virt, virt)] = t_ss
        λbb[np.ix_(occ, occ, virt, virt)] = t_ss
        λab[np.ix_(occ, occ, virt, virt)] = t_os

        # 2. oooo blocks from quadratic terms
        #   λ_{ijkl} ~ 1/2 Σ_{ab} t_{ij}^{ab} t_{kl}^{ab}
        λaa[np.ix_(occ, occ, occ, occ)] += 0.5 * np.einsum(
            "ijab,klab->ijkl", t_ss, t_ss, optimize=True
        )
        λbb[np.ix_(occ, occ, occ, occ)] += 0.5 * np.einsum(
            "ijab,klab->ijkl", t_ss, t_ss, optimize=True
        )
        λab[np.ix_(occ, occ, occ, occ)] += 0.5 * np.einsum(
            "ijab,klab->ijkl", t_os, t_os, optimize=True
        )

        # 3. vvvv blocks from quadratic terms
        #   λ_{abcd} ~ 1/2 Σ_{ij} t_{ij}^{ab} t_{ij}^{cd}
        λaa[np.ix_(virt, virt, virt, virt)] += 0.5 * np.einsum(
            "ijab,ijcd->abcd", t_ss, t_ss, optimize=True
        )
        λbb[np.ix_(virt, virt, virt, virt)] += 0.5 * np.einsum(
            "ijab,ijcd->abcd", t_ss, t_ss, optimize=True
        )
        λab[np.ix_(virt, virt, virt, virt)] += 0.5 * np.einsum(
            "ijab,ijcd->abcd", t_os, t_os, optimize=True
        )

        # 4. ovvv (i,a,j,b) blocks from quadratic terms
        #   X_{iajb} = - Σ_{kc} t_{ik}^{ac} t_{jk}^{bc}
        X_ss = -np.einsum("ikac,jkbc->iajb", t_ss, t_ss, optimize=True)
        X_os = -np.einsum("ikac,jkbc->iajb", t_os, t_os, optimize=True)

        # Embed into λ[p,q,r,s] with p=i, q=a, r=j, s=b positions
        λaa[np.ix_(occ, virt, occ, virt)] += X_ss
        λbb[np.ix_(occ, virt, occ, virt)] += X_ss
        λab[np.ix_(occ, virt, occ, virt)] += X_os

        # antisymmetrize the aa and bb cumulants
        λaa = _antisymmetrize_pairwise(λaa)
        λbb = _antisymmetrize_pairwise(λbb)

        return λaa, λab, λbb, γa, γb, Γ1
