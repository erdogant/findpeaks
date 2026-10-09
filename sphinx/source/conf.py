# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'findpeaks'
copyright = '2026, Erdogan'
author = 'Erdogan'
release = '2.7.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_show_sourcelink = False 


html_context = {
	"default_mode": "dark",
	"doc_path":"sphinx",
}

# -- Set path for autofunctions and autoset version info
#
pkg_path  = str(Path('../..').resolve())
print('package location:', pkg_path)
sys.path.insert(0,pkg_path) 

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.napoleon', # for numpy in autodoc
	'sphinx.ext.autodoc',  # for automatic documentation of docstrings
	'sphinx.ext.mathjax',  # for latex support in html 
	'sphinx_copybutton',   # allow code copy button in examples
	'sphinx_design',	   # enable design elements like grids
	'sphinxcontrib.mermaid'# enable use of mermaid flowcharts in docs
]
language = "en"

templates_path = ['_templates']
