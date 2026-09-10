# Spin-free DSRG-MRPT3 contraction kernels

`forte2/dsrg/dsrg_mrpt3_kernels.py` is generated, not written by hand.

The working equations live once, in composite hole/particle/general indices, in
`terms.py` -- transcribed from forte's spin-adapted `sadsrg_comm.cc`. `expand.py`
expands each term over the elementary core/active/virtual blocks and drops the
combinations that cannot contribute: an amplitude block outside its
hole-particle support, an output block that is not off-diagonal, and, for the
`_od` kernel variants, any operand block a commutator cannot occupy.

To change the equations, edit `terms.py` and regenerate:

    python codegen/gen_kernels.py

`kernel_header.py` is the hand-written part -- the class, the adaptive `einsum`
wrapper, and `_df`, which rebuilds an integral block with three or more virtual
indices from the three-index factors rather than storing it.

## Checking a change

`tests/dsrg/test_dsrg_mrpt3.py` pins the four stage energies separately, both
reference-relaxation trajectories, and equality with the two-component code
under a GHF reference, where the two formalisms coincide. Regenerating and
running those is the check on a change to the equations.

The expansion was originally validated against a composite-index implementation
of the same equations, kernel by kernel on random operands to machine
precision. That reference has been removed now the generated kernels are
established; if the pruning rules in `expand.py` are changed substantially, it
is worth reinstating something equivalent, because the energy tests show a
shifted total rather than naming the kernel at fault.
