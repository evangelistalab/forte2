Quickstart
==========

To get started with Forte2, you can either download it from conda-forge, or compile it from source.

Install with conda
------------------
If you have a conda distribution installed (e.g., `miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_),
you can obtain Forte2 from conda-forge::

    $ conda install -c conda-forge forte2

Compile from source
-------------------
If your platform is not supported by the current conda-forge Forte2 package, 
or if you want to develop Forte2, follow these steps to compile Forte2 from source:

1. Clone the Forte2 repository from GitHub::

   $ git clone git@github.com:evangelistalab/forte2.git
   $ cd forte2

2. Install the required dependencies in ``environment.yml``.
   You can most easily do this in a new conda environment::

    $ conda env create -f environment.yml
    $ conda activate forte2

   The environment uses OpenBLAS by default. To switch to another implementation, run the following command::

    $ conda install "libblas=*=*BLAS_VARIANT"

   Replace ``BLAS_VARIANT`` with ``newaccelerate`` for Apple's Accelerate framework (macOS only) or ``mkl`` for Intel MKL (x86-64 only).
   These tend to be more optimized for the target architectures and are recommended if available.

   Forte2 links the BLAS and LAPACK libraries that the environment selects, which numpy and scipy also use.
   If you switch after you build Forte2, build it again.

3. Build the project::

    $ pip install --no-build-isolation -ve .

4. Run the quick tests::

    $ pytest -m "not slow"
