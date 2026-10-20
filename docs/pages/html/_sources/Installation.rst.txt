
Quickstart
''''''''''

This section provides a quick introduction to using the findpeaks library for peak detection and analysis. The example demonstrates the basic workflow from data import to visualization using :func:`findpeaks.findpeaks.findpeaks.fit`, :func:`findpeaks.findpeaks.findpeaks.plot`, and :func:`findpeaks.findpeaks.findpeaks.plot_persistence`.

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    from findpeaks import findpeaks

    # Initialize with topology method (most robust)
    fp = findpeaks(method='topology')

    # Example data:
    X = fp.import_example('1dpeaks')

    # Peak detection
    results = fp.fit(X)

    # Plot results
    fp.plot()

    # Plot persistence diagram
    fp.plot_persistence()

Installation
''''''''''''''

This section covers the installation process for the findpeaks library, including environment setup and package management.

Create Environment
------------------

For optimal performance and to avoid dependency conflicts, it's recommended to install ``findpeaks`` in an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_findpeaks python=3.6
    conda activate env_findpeaks


Install via ``pip``:

.. code-block:: console

    # Installation from PyPI (recommended):
    pip install findpeaks

    # Install directly from GitHub (unstable version but it is the latest development version):
    pip install git+https://github.com/erdogant/findpeaks


Uninstalling
''''''''''''''

If you want to remove your ``findpeaks`` installation and clean up your environment, follow these steps:

.. code-block:: console

   # Remove findpeaks package
   pip uninstall findpeaks

   # Deactivate the conda environment
   conda deactivate

   # List all environments to verify
   conda env list

   # Remove the findpeaks environment
   conda env remove --name env_findpeaks

   # Verify environment removal
   conda env list



.. include:: add_bottom.add