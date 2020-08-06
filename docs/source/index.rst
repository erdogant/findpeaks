findpeaks's documentation!
==========================

*findpeaks* is Python package for the detection of peaks and valleys in a 1d-vector and 2d-array (images).
Peaks and valleys can be detected using **topology**, **mask**, and the **peakdetect** approach. In addition to peak-detection, various functions 
are readily available for pre-processing the data (denoising, normalizing, resizing), and vizualizing the data (3d-mesh, persistence)


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
  :caption: Use-cases

  Use-cases


.. toctree::
  :maxdepth: 1
  :caption: Code Documentation
  
  Coding quality
  findpeaks.findpeaks



Quick install
-------------

.. code-block:: console

   pip install findpeaks




Source code and issue tracker
------------------------------

Available on Github, `erdogant/findpeaks <https://github.com/erdogant/findpeaks/>`_.
Please report bugs, issues and feature extensions there.

Citing *findpeaks*
------------------
Here is an example BibTeX entry:

@misc{erdogant2019findpeaks,
  title={findpeaks},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/findpeaks}}}



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
