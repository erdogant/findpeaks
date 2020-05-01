.. _code_directive:

-------------------------------------

Quickstart
''''''''''

A quick example how to learn a model on a given dataset.


.. code:: python

    # Import library
    import XXX

    # Retrieve URLs of malicous and normal urls:
    X, y = XXX.load_example()

    # Learn model on the data
    model = XXX.fit_transform(X, y, pos_label='bad')

    # Plot the model performance
    results = XXX.plot(model)


Installation
''''''''''''

Create environment
------------------


If desired, install ``XXX`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_XXX python=3.6
    conda activate env_XXX


Install via ``pip``:

.. code-block:: console

    # The installation from pypi is disabled:
    pip install XXX

    # Install directly from github
    pip install git+https://github.com/erdogant/XXX


Uninstalling
''''''''''''

If you want to remove your ``XXX`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. XXX should be listed.
   conda env list

   # Remove the XXX environment
   conda env remove --name XXX

   # List all the active environments. XXX should be absent.
   conda env list
