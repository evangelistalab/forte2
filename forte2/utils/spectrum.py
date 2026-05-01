import numpy as np
from typing import Literal
from numpy.typing import NDArray
from forte2.data import EH_TO_EV


def convolution(
    vte: list[float] | NDArray[np.floating],
    fosc: list[float] | NDArray[np.floating],
    g: float = 0.2,
    bounds: tuple[float, float] | None = None,
    convolution_type: Literal["lorentzian", "gaussian"] = "lorentzian",
    npts: int = 10000,
    return_as_components: bool = False,
    normalize: bool = True,
):
    """
    Calculate spectrum convolution for pairs of vertical transition energy and oscillator strengths.

    Parameters
    ----------
    vte : list[float]
        The vertical transition energies.
    fosc : list[float]
        The oscillator strengths.
    g : float
        The convolution broadening factor.
    bounds : tuple[float, float] or None
        The energy range for which to generate the spectrum convolution.
        If None, use ``(min(vte) - 4*g, max(vte) + 4*g)``.
    convolution_type : {"lorentzian", "gaussian"}
        The convolution algorithm to use, lorentzian or gaussian.
    npts : int
        The number of points along the energy range.
    return_as_components : bool
        If True, return the individual convolution components for each
        vertical transition energy / oscillator strength pair as rows in a
        two-dimensional array with shape ``(npts, len(vte))``. If False,
        return the summed spectrum as a one-dimensional array with shape
        ``(npts,)``.
    normalize : bool
        If True, normalize convolution curve.

    Returns
    -------
    x_axis : NDArray
        The spectrum convolution energy range.
    spectrum : NDArray
        The spectrum convolution.
    """

    vte_arr = np.asarray(vte, dtype=float) * EH_TO_EV
    fosc_arr = np.asarray(fosc, dtype=float)

    try:
        if g <= 0:
            raise ValueError("The broadening parameter g must be a positive number.")
    except TypeError as exc:
        raise ValueError(
            "The broadening parameter g must be a positive number."
        ) from exc

    if vte_arr.size == 0 or fosc_arr.size == 0:
        raise ValueError("vte and fosc must not be empty.")

    if vte_arr.ndim != 1 or fosc_arr.ndim != 1:
        raise ValueError("vte and fosc must be one-dimensional lists of floats.")

    if type(npts) is not int or npts < 2:
        raise ValueError("npts must be an integer greater than or equal to 2.")

    if vte_arr.shape != fosc_arr.shape:
        raise ValueError(
            f"Parameters vte and fosc must have same shape, got {vte_arr.shape} and {fosc_arr.shape}."
        )

    thrs = 1e-4
    if (np.max(fosc_arr) < thrs) and normalize:
        raise ValueError(
            f"Cannot normalize convolution, no oscillator strength with magnitude greater than {thrs} found."
        )

    if bounds is None:
        b = (vte_arr.min() - 4 * g, vte_arr.max() + 4 * g)
    else:
        b = bounds

    if len(b) != 2 or b[0] >= b[1]:
        raise ValueError(
            f"bounds must be a pair (lower, upper) with lower < upper, got {b}."
        )

    x_axis = np.linspace(b[0], b[1], npts)

    if convolution_type == "lorentzian":
        spectrum = (
            fosc_arr[:, np.newaxis]
            * (0.5 * g / np.pi)
            / (0.25 * g**2 + (x_axis - vte_arr[:, np.newaxis]) ** 2)
        )

    elif convolution_type == "gaussian":
        spectrum = (
            fosc_arr[:, np.newaxis]
            * np.exp(-((x_axis - vte_arr[:, np.newaxis]) ** 2) / (2 * g**2))
            / (g * np.sqrt(2 * np.pi))
        )
    else:
        raise ValueError(
            f"Invalid convolution algorithm {convolution_type!r}. Expected 'lorentzian' or 'gaussian'."
        )

    if return_as_components:
        spectrum = spectrum.T
    else:
        spectrum = spectrum.sum(axis=0)

    if normalize:
        spectrum = spectrum / np.max(spectrum)

    return x_axis, spectrum


if __name__ == "__main__":
    from forte2.system import System
    from forte2.scf import RHF
    from forte2.ci import CISolver
    from forte2.state import State
    from forte2.mcopt import MCOptimizer
    import matplotlib.pyplot as plt

    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    gs = State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[2])
    cex = State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[1])

    rhf = RHF(charge=0)(system)
    ci_solver = CISolver(
        nroots=[1, 4],
        weights=[[0], [1, 1, 1, 1]],
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        states=[gs, cex],
    )
    mc = MCOptimizer(ci_solver, active_frozen_orbitals=[0], do_transition_dipole=True)(
        rhf
    )
    mc.run()

    # select transitions properties between state 0 and states 1, 2, 3, and 4
    vte = list(ci_solver.vertical_transition_energies.values())[1:5]
    fosc = list(ci_solver.oscillator_strengths.values())[1:5]

    for a in [True, False]:
        for b in [True, False]:
            spectrum_range, spectrum_convolution = convolution(
                vte, fosc, return_as_components=a, normalize=b
            )

            plt.plot(spectrum_range, spectrum_convolution)
            plt.savefig(
                f"spectrum_convolution_H2O_return_as_components_{a}_normalize_{b}.pdf",
                bbox_inches="tight",
            )
            plt.show()
