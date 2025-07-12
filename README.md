# forte2

[![Build Status](https://github.com/evangelistalab/forte2/actions/workflows/build.yml/badge.svg)](https://github.com/evangelistalab/forte2/actions/workflows/build.yml)

Compile with `pip install --no-build-isolation -ve .`

## Documentation
In the root directory, run:
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
sphinx-apidoc -o docs/source forte2/.
cd docs
make html
```