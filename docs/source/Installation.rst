
Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    from findpeaks import findpeaks

    # Initialized
    fp = findpeaks(method='topology')

    # Example data:
    X = fp.import_example('1dpeaks')

    # Peak detection
    results = fp.fit(X)

    # Plot
    fp.plot()

    # Plot
    fp.plot_persistence()

Installation
''''''''''''

Create environment
------------------


If desired, install ``findpeaks`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_findpeaks python=3.6
    conda activate env_findpeaks


Install via ``pip``:

.. code-block:: console

    # Installation from pypi:
    pip install findpeaks

    # Install directly from github
    pip install git+https://github.com/erdogant/findpeaks


Uninstalling
''''''''''''

If you want to remove your ``findpeaks`` installation with your environment, it can be as following:

.. code-block:: console

   # Removing findpeaks.
   pip uninstall findpeaks

   # Step out the environments.
   conda deactivate

   # List all the active environments. findpeaks should be listed.
   conda env list

   # Remove the findpeaks environment
   conda env remove --name env_findpeaks

   # List all the active environments. findpeaks should be absent.
   conda env list



.. include:: add_bottom.add