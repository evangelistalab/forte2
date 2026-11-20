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

3. Build the project::

    $ pip install --no-build-isolation -ve .

4. Run the quick tests::

    $ pytest -m "not slow"

.. _install-block2:

Install block2 for DMRG
-----------------------
The density matrix renormalization group (DMRG) solver, ``DMRGSolver``, runs on
`block2 <https://github.com/block-hczhai/block2-preview>`_, which isn't installed with Forte2.
If you want to use DMRG, install block2 0.5.4 in the same environment as Forte2, either
from PyPI or from source.

Install from PyPI
~~~~~~~~~~~~~~~~~
The block2 package on PyPI bundles its own OpenMP runtime and, on Linux, depends on Intel MKL.
Depending on your platform, it needs one more step after you install it::

    $ pip install "block2==0.5.4"

* On macOS, the bundled OpenMP runtime stops the process with ``OMP: Error #15`` because Forte2
  has already loaded the environment's runtime. Replace the bundled copy with a link to the
  environment's copy::

    $ PYBLOCK2_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('pyblock2').origin))")
    $ ln -sf "$CONDA_PREFIX/lib/libomp.dylib" "$PYBLOCK2_DIR/.dylibs/libomp.dylib"

  Reinstalling or upgrading block2 restores the bundled copy, so repeat this step afterward.

* On Linux with an AMD CPU, MKL stops the process with ``Intel MKL FATAL ERROR``. Build a small
  library that reports the CPU as an Intel one, and preload it whenever the environment is
  active::

    $ echo 'int mkl_serv_intel_cpu_true() { return 1; }' > fixcpu.c
    $ cc -shared -fPIC -o "$CONDA_PREFIX/lib/libfixcpu.so" fixcpu.c
    $ conda env config vars set LD_PRELOAD="$CONDA_PREFIX/lib/libfixcpu.so"

  Reactivate the environment for the setting to take effect.

Build from source
~~~~~~~~~~~~~~~~~
A source build links block2 against the BLAS and OpenMP libraries of the Forte2 environment, so it
needs no further steps. The build takes 10-30 minutes and about 1.2 GB of memory per parallel
job.

To build block2 in the active Forte2 environment:

1. Install the build tools::

    $ conda install -c conda-forge cmake cxx-compiler pybind11

2. Clone block2::

    $ git clone --depth 1 -b v0.5.4 https://github.com/block-hczhai/block2-preview.git
    $ cd block2-preview

3. On macOS, select the environment's compilers, which Forte2 is also built with::

    $ export CC=$CONDA_PREFIX/bin/clang CXX=$CONDA_PREFIX/bin/clang++

4. Configure the build. On Linux, replace ``.dylib`` with ``.so``::

    $ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
        -DUSE_MKL=OFF -DF77UNDERSCORE=ON \
        -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libblas.dylib \
        -DLAPACK_LIBRARIES=$CONDA_PREFIX/lib/liblapack.dylib \
        -DBUILD_LIB=ON -DBUILD_EXE=OFF -DLARGE_BOND=ON -DUSE_KSYMM=OFF \
        -DUSE_SU2SZ=ON -DUSE_SG=ON -DUSE_COMPLEX=ON -DUSE_SANY=OFF -DUSE_SINGLE_PREC=OFF \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=$PWD/build/lib

5. Build block2::

    $ cmake --build build --parallel 4

6. Add the compiled module and the ``pyblock2`` package to the environment's Python path::

    $ SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    $ printf '%s\n' "$PWD/build/lib" "$PWD" > "$SITE_PACKAGES/block2.pth"

7. Check that Python finds block2::

    $ python -c "from pyblock2.driver.core import DMRGDriver"

Python reads ``pyblock2`` directly from the clone, so rebuild block2 whenever you update the clone.
