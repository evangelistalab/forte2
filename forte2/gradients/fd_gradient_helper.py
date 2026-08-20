import numpy as np

# npoints -> (offsets, weights, denominator).
# The derivative is sum(weights * f(x + offsets * h)) / (denominator * h).
# Each stencil is central, so the reference point itself is never evaluated, and
# the leading error term is O(h**npoints).
_CENTRAL_STENCILS = {
    2: ((-1.0, 1.0), (-1.0, 1.0), 2.0),
    4: ((-2.0, -1.0, 1.0, 2.0), (1.0, -8.0, 8.0, -1.0), 12.0),
    6: (
        (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0),
        (-1.0, 9.0, -45.0, 45.0, -9.0, 1.0),
        60.0,
    ),
}


def central_stencil(npoints):
    """
    Return the central-difference stencil with `npoints` points.

    Parameters
    ----------
    npoints : int
        Number of stencil points. One of 2, 4, or 6.

    Returns
    -------
    tuple[tuple[float], tuple[float], float]
        The offsets (in units of the step), the weights, and the denominator.
        The derivative is
        ``sum(weights * f(x + offsets * step)) / (denominator * step)``.

    Raises
    ------
    ValueError
        If `npoints` is not a supported stencil size.
    """
    if npoints not in _CENTRAL_STENCILS:
        raise ValueError(
            f"npoints must be one of {sorted(_CENTRAL_STENCILS)}, but got {npoints}."
        )
    return _CENTRAL_STENCILS[npoints]


def finite_difference(f, x, *, step=1.0e-3, npoints=4, components=None, progress=None):
    r"""
    Differentiate `f` at `x` by central finite differences.

    `f` may return a scalar or an array; the derivative preserves that shape.
    `x` may be a scalar or an array of any shape, and `f` receives displaced
    values of the same shape. `x` itself is never modified.

    The number of evaluations of `f` is ``npoints`` times the number of
    differentiated components.
    The truncation error is ``step**npoints`` while noise in `f` enters as ``noise / step``.

    Parameters
    ----------
    f : Callable
        The function to differentiate.
    x : float | array_like
        The point at which to differentiate.
    step : float, optional, default=1.0e-3
        The displacement, in the units of `x`.
    npoints : int, optional, default=4
        Central-difference stencil size; see :func:`central_stencil`.
    components : Sequence, optional
        Which entries of `x` to differentiate, as indices into `x` (an ``int``
        for a 1-D `x`, a tuple for higher rank). If None, every entry is
        differentiated. Must be None when `x` is a scalar.
    progress : Callable, optional
        Called as ``progress(done, total)`` after each evaluation of `f`, where
        `total` is the number of evaluations the call will make. Intended for
        progress reporting on long runs.

    Returns
    -------
    NDArray | float
        If `x` is a scalar, the derivative, with the shape of ``f(x)``.
        If `components` is None, an array of shape ``x.shape + f(x).shape``.
        Otherwise an array of shape ``(len(components),) + f(x).shape``.

    Raises
    ------
    ValueError
        If `step` is not positive, if `npoints` is unsupported, if `components`
        is given for a scalar `x`, or if `f` returns inconsistent shapes.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(finite_difference(np.sin, 0.0)), 8)
    1.0
    >>> finite_difference(lambda v: v @ v, np.array([1.0, 2.0])).round(6)
    array([2., 4.])
    """
    if not np.isscalar(step) or step <= 0.0:
        raise ValueError(f"step must be a positive number, but got {step}.")
    offsets, weights, denominator = central_stencil(npoints)
    scale = 1.0 / (denominator * step)

    x_is_scalar = np.isscalar(x) or (np.ndim(x) == 0)
    if x_is_scalar:
        if components is not None:
            raise ValueError("components cannot be given when x is a scalar.")
        indices = None
        x_array = float(x)
    else:
        x_array = np.array(x, dtype=float)  # a copy: x is never modified
        indices = (
            list(np.ndindex(x_array.shape)) if components is None else list(components)
        )

    total = npoints * (1 if x_is_scalar else len(indices))
    done = 0

    def evaluate(displaced):
        nonlocal done
        value = np.asarray(f(displaced), dtype=float)
        done += 1
        if progress is not None:
            progress(done, total)
        return value

    def differentiate(displace):
        """Weighted sum of f over the stencil, given a displacement callable."""
        derivative = None
        for offset, weight in zip(offsets, weights):
            value = evaluate(displace(offset * step))
            if derivative is None:  # record reference shape in first iteration
                derivative = weight * value
                reference_shape = value.shape
            elif value.shape != reference_shape:
                raise ValueError(
                    f"f returned shape {value.shape} at one displacement and "
                    f"{reference_shape} at another; the shape must not depend on x."
                )
            else:
                derivative += weight * value
        return derivative * scale

    if x_is_scalar:
        return differentiate(lambda delta: x_array + delta)

    derivatives = []
    for index in indices:

        def displace(delta, index=index):
            displaced = x_array.copy()
            displaced[index] += delta
            return displaced

        derivatives.append(differentiate(displace))

    result = np.stack(derivatives)
    if components is None:
        result = result.reshape(x_array.shape + result.shape[1:])
    return result
