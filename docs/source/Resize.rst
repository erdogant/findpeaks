.. _code_directive:

-------------------------------------

Resize
''''''''''''


AAA

.. code:: python

    # Import library
    import findpeaks

    # Load example data set    
    X,y_true = findpeaks.load_example()

    # Retrieve URLs of malicous and normal urls:
    model = findpeaks.fit_transform(X, y_true, pos_label='bad', train_test=True, gridsearch=True)

    # The test error will be shown
    results = findpeaks.plot(model)


