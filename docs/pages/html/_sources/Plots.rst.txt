
One-dimensional plots
-------------------------------------

Pre-processing
'''''''''''''''
The pre-processing in a 1d-vector is based on the interpolation: function: :func:`findpeaks.interpolate.interpolate_line1d`

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize
    fp = findpeaks(method='topology', interpolate=10)
    # Import example
    X = fp.import_example("1dpeaks")
    # Detect peaks
    results = fp.fit(X)
    # Plot
    fp.plot()

.. |figP4| image:: ../figs/1dpeaks_interpolate_original.png
.. |figP5| image:: ../figs/1dpeaks_interpolate.png

.. table:: Inerpolation
   :align: center

   +----------+----------+
   | |figP4|  | |figP5|  |
   +----------+----------+


Persistence
''''''''''''

The persistence plot is called with the function: :func:`findpeaks.findpeaks.findpeaks.plot_persistence`, and provides two plots.
The left is the detected peaks with the ranking of the peaks (1=best), and the right plot the homology-persitence plot. See section topology for more details.

.. code:: python

    # Plot
    fp.plot_persistence()


.. |figP6| image:: ../figs/1d_plot_persistence.png

.. table:: Persistence Plot
   :align: center

   +----------+
   | |figP6|  |
   +----------+
   
   
Two-dimensional plots
-------------------------------------

Pre-processing plot
'''''''''''''''''''''
The pre-processing plot is developed for 2D arrays (images) only: function: :func:`findpeaks.findpeaks.findpeaks.plot_preprocessing`
Depending on the number of user defined pre-processing steps, the plot will add new subplots.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize
    fp = findpeaks(method='topology', whitelist=['peak'])
    # Import example
    X = fp.import_example("2dpeaks")
    # Detect peaks
    results = fp.fit(X)
    # Plot
    fp.plot_preprocessing()


.. |figP0| image:: ../figs/plot_example_norm.png

.. table:: Preprocessing plot
   :align: center

   +----------+
   | |figP0|  |
   +----------+
   

Plot
''''''''''''

The **plot** function :func:`findpeaks.findpeaks.findpeaks.plot` plots the 3 major steps: 
    * input data
    * final pre-processed image 
    * peak detection.

.. code:: python

    # Plot
    fp.plot(figure_order='horizontal')


.. |figP1| image:: ../figs/plot_example1.png

.. table:: Final results
   :align: center

   +----------+
   | |figP1|  |
   +----------+
   

Persistence plot
''''''''''''''''''

The persistence plot is called with the function: :func:`findpeaks.findpeaks.findpeaks.plot_persistence`, and provides two plots.
The left is the detected peaks with the ranking of the peaks (1=best), and the right plot the homology-persitence plot. See section topology for more details.

.. code:: python

    # Plot
    fp.plot_persistence()


.. |figP2| image:: ../figs/plot_persistence.png

.. table:: Persistence Plot
   :align: center

   +----------+
   | |figP2|  |
   +----------+


3D-mesh
''''''''''''

The mesh plot can easily be created using the function: :func:`findpeaks.findpeaks.findpeaks.plot_mesh`.
It converts the two image into a 3d mesh plot.

.. code:: python

    # Plot
    fp.plot_mesh()
    # Rotate to make a top view
    fp.plot_mesh(view=(90,0))


.. |figP7| image:: ../figs/2dpeaks_mesh1_norm.png
.. |figP8| image:: ../figs/2dpeaks_mesh2_norm.png
.. |figP9| image:: ../figs/2dpeaks_mesh3_norm.png
.. |figP10| image:: ../figs/2dpeaks_mesh4_norm.png

.. table:: Mesh plot. Top: 3D mesh. Bottom: top view.
   :align: center

   +----------+----------+
   | |figP7|  | |figP8|  |
   +----------+----------+
   | |figP9|  | |figP10| |
   +----------+----------+




.. include:: add_bottom.add