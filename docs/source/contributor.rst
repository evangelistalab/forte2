Contributor Guide
=================

Code Standards
--------------
Forte2 uses the `functional composition <https://en.wikipedia.org/wiki/Function_composition_(computer_science)>`_ style (like the `TensorFlow functional API <https://www.tensorflow.org/guide/keras/functional_api>`_) for most quantum chemical methods, with the following programmatic flow:

>>> rhf = forte2.scf.RHF(charge=0)(system)
>>> ci = forte2.ci.CI(states=state, active_orbitals=[...], ...)(rhf)
>>> ci.run()
-0.75102385

This means that most methods should be a class with a ``__call__`` method that takes the previous method as an argument, and returns a new instance. Each method should also have a ``run`` method that executes the method and returns the result. This allows for easy chaining of methods, as shown in the example above.

If you create new C++ functions or classes that are exposed through nanobind, make sure to document them in the binding code (`forte2/api`), 
and run the `nanobind stubgen <https://nanobind.readthedocs.io/en/latest/typing.html#command-line-interface>`_ in the root directory::

    python -m nanobind.stubgen -m forte2._forte2 -O forte2 -r

This provides "stub files" for (1) the RTD documentation, (2) autocomplete, argument hints, etc in IDEs like VSCode. Make sure to commit the stub files along with your other changes.

Style Guide
-----------
Forte2 uses ``Black Formatter`` for Python code and ``Clang-Format`` for C++ code. Both can be obtained in VSCode via the extensions marketplace or via the command line.

Docstrings should follow the `Numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
For example::
    class MagicFCI:
    r"""
    A class that runs the full configuration interaction in constant time.

    .. math::
        |\Psi\rangle = \sum_{i=1}^{N} c_i |\Phi_i\rangle

    Parameters
    ----------
    system : forte2.system.System
        The system to perform the configuration interaction on.
    ci_dim : int
        The dimension of the configuration interaction space.
    algorithm : string, optional, default="magic"
        The algorithm to use for the configuration interaction. Options are "magic", "brute_force", and "magic_brute_force".
    shift : float, optional
        The eigenvalue shift :math:`\eta` to use. If None, no shift is applied.

    Attributes
    ----------
    ci_vec : NDArray
        The configuration interaction vector.

    Returns
    -------
    float
        The energy of the system after running the configuration interaction.

    Notes
    -----
    The "magic" algorithm follows the approach described in J. Magic Theory Comput., 1, 1324 (2036).

    Raises
    ------
    ValueError
        If the provided system is not in the training set of OpenMol.

    Examples
    --------
    >>> fci = forte2.ci.MagicFCI(system, ci_dim=100, algorithm="magic", shift=0.1)(rhf)
    >>> fci.run()
    -42.0000000000
    """

A few notes on the docstring style (for more, see the Numpy style guide linked above):

- Classes should have a docstring directly after the class definition, and not under the ``__init__`` method.
- The first line should be a brief summary of the class or function.
- Display-style LaTeX equations can be included using the ``.. math::`` directive, note the indentation and newlines before and after the equation. If you use LaTeX, the docstring should be a raw string (i.e., prefixed with ``r"""``). Inline equations can be included like ``:math:\`\eta\```.
- The ``Parameters`` section should include all parameters (arguments for functions, init arguments for classes), their types, and whether they are optional, with default values if applicable. If the default value is ``None`` or other placeholder (*i.e.* not used as a value), then only ``optional`` is needed.
- The ``Attributes`` section should include public attributes of the class, their types, and a brief description.
- The return should have the type but not necessarily the name.
- Only public methods and functions *need* to be documented (all functions with a leading underscore, like ``_my_private_function`` are considered private), however private functions should still be well documented for clarity.