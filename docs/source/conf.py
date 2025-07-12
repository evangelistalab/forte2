import os
import sys

# Add the project root to the system path
sys.path.insert(0, os.path.abspath("../"))

project = "forte2"
copyright = "2025, Evangelista Lab"
author = "Evangelista Lab"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # for Google/NumPy style docstrings
    "sphinx_autodoc_typehints",  # for type hints
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
