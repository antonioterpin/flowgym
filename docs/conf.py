"""Sphinx configuration for the FlowGym documentation site."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "FlowGym"
copyright = "2026, FlowGym contributors"
author = "FlowGym contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["build", "source", "Thumbs.db", ".DS_Store"]

root_doc = "index"
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

myst_enable_extensions = [
    "colon_fence",
]
myst_heading_anchors = 3

autodoc_mock_imports = [
    "cv2",
    "ipywidgets",
    "openpiv",
    "pyoptflow",
    "robo_goggles",
    "skimage",
    "synthpix",
    "torch",
    "torchvision",
    "wandb",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
add_module_names = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
