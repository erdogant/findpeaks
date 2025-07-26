findpeaks's documentation!
==========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|


*findpeaks* is a comprehensive Python library for robust detection and analysis of peaks and valleys in both 1D vectors and 2D arrays (images). The library provides multiple detection algorithms including **topology-based persistent homology** (most robust), **mask-based local maximum filtering**, and traditional **peakdetect** approaches. 

**Key Features:**
    - **Multiple Detection Methods**: Topology (persistent homology)
    - **1D and 2D Support**: Works with time series, signals, images, and spatial data
    - **Advanced Preprocessing**: Denoising, scaling, interpolation, and image preprocessing
    - **Rich Visualization**: Persistence diagrams, 3D mesh plots, preprocessing steps, and masking plots
    - **Mathematical Stability**: Topology method provides mathematically grounded peak detection
    - **Hough Transform Applications**: Enhanced robustness for computer vision applications

The library includes comprehensive preprocessing capabilities (denoising, normalizing, resizing) and advanced visualization tools (3D mesh plots, persistence diagrams, preprocessing pipelines) to help users understand and analyze their data effectively.


-----------------------------------

.. note::
	`Medium Blog: A Step-by-Step Guide To Accurately Detect Peaks and Valleys. <https://erdogant.medium.com>`_

-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/findpeaks/pages/html/Documentation.html>`_ in various ways, have a look at the `sponsor page <https://erdogant.github.io/findpeaks/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/findpeaks/>`_ page.

	.. code-block:: console

	   pip install findpeaks

-----------------------------------


Content
=========

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation


.. toctree::
  :maxdepth: 2
  :caption: Algorithms

  Topology
  Mask
  Peakdetect
  Caerus
  Performance

.. toctree::
  :maxdepth: 1
  :caption: Pre-processing

  Pre-processing
  Denoise


.. toctree::
  :maxdepth: 1
  :caption: Plots

  Plots


.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples
  Use-cases


.. toctree::
  :maxdepth: 1
  :caption: Documentation

  Documentation
  Function_Reference
  Coding quality




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |python| image:: https://img.shields.io/pypi/pyversions/findpeaks.svg
    :alt: |Python
    :target: https://erdogant.github.io/findpeaks/

.. |pypi| image:: https://img.shields.io/pypi/v/findpeaks.svg
    :alt: |Python Version
    :target: https://pypi.org/project/findpeaks/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/findpeaks/

.. |stars| image:: https://img.shields.io/github/stars/erdogant/findpeaks
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/findpeaks

.. |LOC| image:: https://sloc.xyz/github/erdogant/findpeaks/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/findpeaks

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/findpeaks?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/findpeaks

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/findpeaks?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/findpeaks

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/findpeaks/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/findpeaks.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/findpeaks/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/findpeaks.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/findpeaks/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/findpeaks/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/findpeaks/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/findpeaks/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/260400472.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/260400472

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/findpeaks
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/findpeaks


.. include:: add_bottom.add
