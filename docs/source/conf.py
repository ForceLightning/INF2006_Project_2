# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "INF2006 Assignment 2"
copyright = "2024, Cheng Yi Xing, Jurgen Tan Yu Teng, Kok Yong En Christopher, Ng Zi Bin, Wong Yok Hung, and the Singapore Institute of Technology"
author = "Cheng Yi Xing, Jurgen Tan Yu Teng, Kok Yong En Christopher, Ng Zi Bin, Wong Yok Hung, and the Singapore Institute of Technology"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src/"))
