# Sphinx configuration for NuclearMatter-Backend (readthedocs-deployable).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "src")))

project = "NuclearMatter-Backend"
author = "R. O'Shaughnessy et al."
copyright = "2026, " + author

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# heavy runtime deps are mocked so the docs build anywhere (incl. readthedocs)
autodoc_mock_imports = ["jax", "jaxlib", "diffrax", "equinox", "h5py"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "NuclearMatter-Backend"
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
