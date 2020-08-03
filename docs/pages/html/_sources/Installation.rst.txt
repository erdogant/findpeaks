.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import findpeaks

    # Retrieve URLs of malicous and normal urls:
    X, y = findpeaks.load_example()

    # Learn model on the data
    model = findpeaks.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = findpeaks.plot(model)


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

    # The installation from pypi is disabled:
    pip install findpeaks

    # Install directly from github
    pip install git+https://github.com/erdogant/findpeaks


Uninstalling
''''''''''''

If you want to remove your ``findpeaks`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. findpeaks should be listed.
   conda env list

   # Remove the findpeaks environment
   conda env remove --name findpeaks

   # List all the active environments. findpeaks should be absent.
   conda env list
