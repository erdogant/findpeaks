# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Download rst file -----------------------------------------------------
try:
	from urllib.request import urlretrieve
	sponsor_url_rst = 'https://erdogant.github.io/docs/rst/sponsor.rst'
	sponsor_file = "sponsor.rst"
	if os.path.isfile(sponsor_file):
		os.remove(sponsor_file)
		print('Update sponsor rst file.')
	urlretrieve (sponsor_url_rst, sponsor_file)
except:
	print('Downloading sponsor.rst file failed.')



# -- Project information -----------------------------------------------------

project = 'findpeaks'
copyright = '2020, Erdogan Taskesen'
author = 'Erdogan Taskesen'

# The master toctree document.
master_doc = 'index'

# The full version, including alpha/beta/rc tags
release = 'findpeaks'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
	"sphinx.ext.intersphinx",
	"sphinx.ext.autosectionlabel",
	"rst2pdf.pdfbuilder",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# autodoc_mock_imports = ['cv2','keras']


pdf_documents = [('index', u'findpeaks', u'findpeaks', u'Erdogan Taskesen'),]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'default'
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['css/custom.css',]

# html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'carbon_ads.html', 'sourcelink.html', 'searchbox.html'] }
