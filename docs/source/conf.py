import os
import sys

# Add the project root to the system path
sys.path.insert(0, os.path.abspath("../"))

project = "Forte2"
copyright = "2025, Evangelista Lab"
author = "Evangelista Lab"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",  # for NumPy style docstrings
    "sphinx_autodoc_typehints",  # for type hints
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 3,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/evangelistalab/forte2",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
}

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "member-order": "bysource",
    "show-inheritance": True,
}

numpydoc_show_class_members = False
