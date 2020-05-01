.. _code_directive:

-------------------------------------

Examples
''''''''''

Learn new model with gridsearch and train-test set
--------------------------------------------------

AAA

.. code:: python

    # Import library
    import XXX

    # Load example data set    
    X,y_true = XXX.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = XXX.fit_transform(X, y_true, pos_label='bad', train_test=True, gridsearch=True)

    # The test error will be shown
    results = XXX.plot(model)


Learn new model on the entire data set
--------------------------------------------------

BBBB


.. code:: python

    # Import library
    import XXX

    # Load example data set    
    X,y_true = XXX.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = XXX.fit_transform(X, y_true, pos_label='bad', train_test=False, gridsearch=True)

    # The train error will be shown. Such results are heavily biased as the model also learned on this set of data
    results = XXX.plot(model)

