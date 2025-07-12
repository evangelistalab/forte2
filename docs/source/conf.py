import os
import sys

# Add the project root to the system path
sys.path.insert(0, os.path.abspath("../"))

project = "Forte2 Documentation"
copyright = "2025, Evangelista Lab"
author = "Evangelista Lab"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",  # for NumPy style docstrings
    "sphinx_autodoc_typehints",  # for type hints
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
autodoc_default_options = {
    "members": True,
    "private-members": False,
    "member-order": "bysource",
    "show-inheritance": True,
}

numpydoc_show_class_members = False