
findpeaks's documentation!
==========================

*findpeaks* is Python package for the detection of peaks and valleys in a 1d-vector and 2d-array (images).
Peaks and valleys can be detected using **topology**, **mask**, and the **peakdetect** approach. In addition to peak-detection, various functions 
are readily available for pre-processing the data (denoising, normalizing, resizing), and vizualizing the data (3d-mesh, persistence)


|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|

.. include:: add_top.add


You contribution is important
==============================
If you ❤️ this project, **star** this repo at the `github page <https://github.com/erdogant/findpeaks/>`_ and have a look at the `sponser page <https://erdogant.github.io/findpeaks/pages/html/Documentation.html>`_!


Github
======
Please report bugs, issues and feature extensions at `github <https://github.com/erdogant/findpeaks/>`_.



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
  Coding quality
  findpeaks.findpeaks




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
