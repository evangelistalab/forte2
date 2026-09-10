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

`tests/dsrg/test_dsrg_mrpt3_kernels.py` contracts random operands through both
the generated kernels and `_DSRGDenseHelper`, which holds the same equations in
composite indices and is kept as ground truth. Every kernel must agree to
machine precision. That check is what catches an expansion or pruning mistake;
the energy tests alone would not localize one.
