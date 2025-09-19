project = "Forte2"
copyright = "2025, Evangelista Lab"
author = "Evangelista Lab"
release = "0.2.0"

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",  # for NumPy style docstrings
    "sphinx_autodoc_typehints",  # for type hints
]
autoapi_dirs = ["../../forte2"]

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
html_logo = "forte2-logo.png"

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    # "show-module-summary",
    # "special-members",
    # "imported-members",
]

autoapi_ignore = ["*/fetch_basis.py", "*/fetch_ccrepo.py"]

numpydoc_show_class_members = False


def skip_private_with_exceptions(app, what, name, obj, skip, options):
    """Skip all private members except those in the exceptions list."""
    short_name = name.split(".")[-1]
    is_private = short_name.startswith("_") and not short_name.endswith("__")
    exceptions = ["_forte2"]
    if (is_private) and (short_name not in exceptions):
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_private_with_exceptions)
