from forte2.ci.ci_utils import (
    spin_free_1rdm,
    spin_free_2rdm,
    spin_free_3rdm,
    make_2cumulant_sf,
    make_3cumulant_sf,
    make_2cumulant_so,
    make_3cumulant_so,
)

_SD_COMPONENTS_BY_ORDER = {
    1: (("g1_a", "g1_b"),),
    2: (("g2_ab", "g2_aa", "g2_bb"),),
    3: (("g3_aab", "g3_abb", "g3_aaa", "g3_bbb"),),
}


class RDMs:
    """
    Container for the reduced density matrices (RDMs) computed by a CI-type solver.

    Stores whichever native building blocks a backend computed for a given bra/ket root
    pair, tagged by ``representation``:

    - ``"sd"`` (spin-dependent): backends that resolve RDMs by spin channel (``CISigmaBuilder``,
      ``SelectedCIHelper``). Components are named ``g{order}_{spin_label}``, e.g. ``g1_a``,
      ``g1_b``, ``g2_aa``, ``g2_ab``, ``g2_bb``, ``g3_aaa``, ``g3_aab``, ``g3_abb``, ``g3_bbb``.
      ``g2_aa``/``g2_bb``/``g3_aaa``/``g3_bbb`` are stored in the packed form the backends return
      natively; ``g2_ab``/``g3_aab``/``g3_abb`` are stored as returned (full or packed-hybrid).
    - ``"so"`` (spin-orbital): backends that work directly in a spinor basis with no spin
      decomposition (``RelCISigmaBuilder``, ``RelSelectedCIHelper``). Components are named
      ``g1``, ``g2``, ``g3`` and are already full tensors.

    Spin-free (for ``"sd"``) or spin-orbital (for ``"so"``) RDMs and cumulants are derived
    lazily from these components via :mod:`forte2.ci.ci_utils` and cached.
    """

    def __init__(self, representation, **components):
        if representation not in ("sd", "so"):
            raise ValueError(
                f"Unknown RDMs representation '{representation}', expected 'sd' or 'so'"
            )
        self.representation = representation
        self._components = components
        self._cache = {}

    def __getattr__(self, name):
        # Allow direct access to native components, e.g. rdms.g2_aa
        try:
            return self._components[name]
        except KeyError:
            raise AttributeError(
                f"'RDMs' object (representation='{self.representation}') has no component "
                f"'{name}'; available components: {sorted(self._components)}"
            ) from None

    def _require(self, order):
        if self.representation == "so":
            names = (f"g{order}",)
        else:
            (names,) = _SD_COMPONENTS_BY_ORDER[order]
        missing = [n for n in names if n not in self._components]
        if missing:
            raise ValueError(
                f"RDMs object (representation='{self.representation}') is missing the "
                f"order-{order} component(s) {missing} needed for this quantity; it was built "
                f"with {sorted(self._components)}"
            )
        return names

    def _rdm(self, order, assemble_sd):
        key = f"rdm{order}"
        if key not in self._cache:
            names = self._require(order)
            if self.representation == "so":
                self._cache[key] = self._components[names[0]]
            else:
                self._cache[key] = assemble_sd(*(self._components[n] for n in names))
        return self._cache[key]

    @property
    def rdm1(self):
        """The 1-RDM: spin-free for 'sd' backends, spin-orbital for 'so' backends."""
        return self._rdm(1, spin_free_1rdm)

    @property
    def rdm2(self):
        """The 2-RDM: spin-free for 'sd' backends, spin-orbital for 'so' backends."""
        return self._rdm(2, spin_free_2rdm)

    @property
    def rdm3(self):
        """The 3-RDM: spin-free for 'sd' backends, spin-orbital for 'so' backends."""
        return self._rdm(3, spin_free_3rdm)

    @property
    def cumulant2(self):
        """The 2-cumulant, derived from ``rdm1``/``rdm2``."""
        if "cumulant2" not in self._cache:
            make_cumulant2 = (
                make_2cumulant_so if self.representation == "so" else make_2cumulant_sf
            )
            self._cache["cumulant2"] = make_cumulant2(self.rdm1, self.rdm2)
        return self._cache["cumulant2"]

    @property
    def cumulant3(self):
        """The 3-cumulant, derived from ``rdm1``/``rdm2``/``rdm3``."""
        if "cumulant3" not in self._cache:
            make_cumulant3 = (
                make_3cumulant_so if self.representation == "so" else make_3cumulant_sf
            )
            self._cache["cumulant3"] = make_cumulant3(self.rdm1, self.rdm2, self.rdm3)
        return self._cache["cumulant3"]
