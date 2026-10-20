
Quick Examples
''''''''''''''''

This section provides comprehensive examples demonstrating the capabilities of the findpeaks library for both 1D and 2D data analysis. Each example showcases different detection methods, preprocessing techniques, and visualization options using functions like :func:`findpeaks.findpeaks.findpeaks.fit`, :func:`findpeaks.findpeaks.findpeaks.plot`, and :func:`findpeaks.findpeaks.findpeaks.plot_persistence`.

1D-vector Analysis
-------------------------------------

The findpeaks library excels at detecting peaks and valleys in 1D data such as time series, signals, and vector data using :func:`findpeaks.findpeaks.findpeaks.peaks1d`. Below are examples demonstrating various detection methods and preprocessing techniques.

Find peaks in low sampled dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates basic peak detection on a small dataset using the default peakdetect method via :func:`findpeaks.peakdetect.peakdetect`. The lookahead parameter is set to 1 for optimal performance on small datasets.

.. code:: python

	# Load library
	from findpeaks import findpeaks
	# Data
	X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
	# Initialize
	fp = findpeaks(lookahead=1)
	results = fp.fit(X)
	# Plot
	fp.plot()

.. image:: ../figs/fig1_raw.png
  :width: 600


Interpolation for Enhanced Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interpolation via :func:`findpeaks.interpolate.interpolate_line1d` can improve peak detection by creating smoother signals. This example shows how interpolation affects the detection results.

.. code:: python

	# Initialize with interpolation parameter
	fp = findpeaks(lookahead=1, interpolate=10)
	results = fp.fit(X)
	fp.plot()


.. image:: ../figs/fig1_interpol.png
  :width: 600


Comparison of Peak Detection Methods (1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example compares the peakdetect method via :func:`findpeaks.peakdetect.peakdetect` and topology method via :func:`findpeaks.stats.topology` on the same dataset, demonstrating the different characteristics of each approach.

.. code:: python

	# Load library
	from findpeaks import findpeaks
	# Data
	X = [10,11,9,23,21,11,45,20,11,12]
	# Initialize
	fp = findpeaks(method='peakdetect', lookahead=1)
	results = fp.fit(X)
	# Plot
	fp.plot()

	fp = findpeaks(method='topology', lookahead=1)
	results = fp.fit(X)
	fp.plot()
	fp.plot_persistence()


.. |ex3| image:: ../figs/fig2_peakdetect.png
.. |ex4| image:: ../figs/fig2_topology.png

.. table:: Comparison of detection methods: peakdetect (left) vs topology (right)
   :align: center

   +----------+----------+
   | |ex3|    | |ex4|    |
   +----------+----------+

.. image:: ../figs/fig2_persistence.png
  :width: 600




Comparison of Peak Detection Methods with Interpolation (2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how interpolation via :func:`findpeaks.interpolate.interpolate_line1d` affects both peakdetect and topology methods, showing the enhanced detection capabilities.

.. code:: python

	# Initialize with interpolate parameter
	fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10)
	results = fp.fit(X)
	fp.plot()

	fp = findpeaks(method='topology', lookahead=1, interpolate=10)
	results = fp.fit(X)
	fp.plot()

.. |ex5| image:: ../figs/fig2_peakdetect_int.png
.. |ex6| image:: ../figs/fig2_topology_int.png

.. table:: Comparison with interpolation: peakdetect (left) vs topology (right)
   :align: center

   +----------+----------+
   | |ex5|    | |ex6|    |
   +----------+----------+



Find peaks in high sampled dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates peak detection on a large, noisy dataset using :func:`findpeaks.findpeaks.findpeaks.plot1d`, showing how different methods handle complex signals with multiple frequency components.

.. code:: python

	# Load library
	import numpy as np
	from findpeaks import findpeaks

	# Data
	i = 10000
	xs = np.linspace(0,3.7*np.pi,i)
	X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

	# Initialize
	fp = findpeaks(method='peakdetect')
	results = fp.fit(X)
	# Plot
	fp.plot1d()

	fp = findpeaks(method='topology', limit=1)
	results = fp.fit(X)
	fp.plot1d()
	fp.plot_persistence()


.. |ex7| image:: ../figs/fig3.png
.. |ex8| image:: ../figs/fig3_topology.png

.. table:: High-sampling comparison: peakdetect (left) vs topology (right)
   :align: center

   +----------+----------+
   | |ex7|    | |ex8|    |
   +----------+----------+

.. image:: ../figs/fig3_persistence_limit.png
  :width: 600




2D-array (Image) Analysis
---------------------------------------------------

The findpeaks library provides robust peak detection capabilities for 2D data including images, spatial data, and matrices using :func:`findpeaks.findpeaks.findpeaks.peaks2d`. The examples below demonstrate various preprocessing techniques and detection methods.

Find peaks using default settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The input image:

.. image:: ../figs/plot_example.png
  :width: 600



.. code:: python

	# Import library
	from findpeaks import findpeaks

	# Import example
	X = fp.import_example()
	print(X)
	# array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.4, 0.4],
	#        [0. , 0. , 0. , 0. , 0. , 0. , 0.7, 1.4, 2.2, 1.8],
	#        [0. , 0. , 0. , 0. , 0. , 1.1, 4. , 6.5, 4.3, 1.8],
	#        [0. , 0. , 0. , 0. , 0. , 1.4, 6.1, 7.2, 3.2, 0.7],
	#        [..., ..., ..., ..., ..., ..., ..., ..., ..., ...],
	#        [0. , 0.4, 2.9, 7.9, 5.4, 1.4, 0.7, 0.4, 1.1, 1.8],
	#        [0. , 0. , 1.8, 5.4, 3.2, 1.8, 4.3, 3.6, 2.9, 6.1],
	#        [0. , 0. , 0.4, 0.7, 0.7, 2.5, 9. , 7.9, 3.6, 7.9],
	#        [0. , 0. , 0. , 0. , 0. , 1.1, 4.7, 4. , 1.4, 2.9],
	#        [0. , 0. , 0. , 0. , 0. , 0.4, 0.7, 0.7, 0.4, 0.4]])

	# Initialize
	fp = findpeaks(method='mask')
	# Fit
	fp.fit(X)

	# Plot the pre-processing steps
	fp.plot_preprocessing()
	# Plot all
	fp.plot()

	# Initialize
	fp = findpeaks(method='topology')
	# Fit
	fp.fit(X)


The masking approach effectively detects the correct peaks in the image data.

.. code:: python

	fp.plot()

.. image:: ../figs/2dpeaks_mask.png
  :width: 600


Conversion from 2D to 3D mesh plots provides excellent visualization capabilities. The surface appears rough due to the low-resolution input data.

.. code:: python

	fp.plot_mesh()

.. |ex9| image:: ../figs/2dpeaks_mesh1.png
.. |ex10| image:: ../figs/2dpeaks_mesh2.png

.. table:: 3D mesh visualization: wireframe (left) and surface (right)
   :align: center

   +----------+----------+
   | |ex9|    | |ex10|   |
   +----------+----------+


The persistence plot demonstrates accurate peak detection with quantitative significance measures.

.. code:: python

	fp.plot_persistence()

.. image:: ../figs/2dpeaks_pers.png
  :width: 600


Find peaks with advanced pre-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This example demonstrates the power of preprocessing techniques in improving peak detection accuracy.

.. code:: python

	# Import library
	from findpeaks import findpeaks

	# Import example
	X = fp.import_example()

	# Initialize with preprocessing parameters
	fp = findpeaks(method='topology', scale=True, denoise=10, togray=True, imsize=(50,100))

	# Fit
	results = fp.fit(X)

	# Plot all
	fp.plot()
	
	# Plot preprocessing
	fp.plot_preprocessing()


.. |ex11| image:: ../figs/2dpeaks_raw.png
.. |ex12| image:: ../figs/2dpeaks_interpolate.png
.. |ex13| image:: ../figs/2dpeaks_raw_processed.png

.. table:: Preprocessing pipeline: raw (left), interpolated (center), processed (right)
   :align: center

   +----------+----------+----------+
   | |ex11|   | |ex12|   |  |ex13|  |
   +----------+----------+----------+


The masking approach may not perform optimally with preprocessing that includes weighted smoothing, which is not ideal for local maximum detection.

.. code:: python
	
	fp.plot()


.. image:: ../figs/2dpeaks_mask_proc.png
  :width: 600


The mesh plot shows higher resolution due to the smoothing effects of preprocessing steps.

.. code:: python
	
	fp.plot_mesh()

.. |ex14| image:: ../figs/2dpeaks_meshs1.png
.. |ex15| image:: ../figs/2dpeaks_meshs2.png

.. table:: Enhanced mesh visualization: wireframe (left) and surface (right)
   :align: center

   +----------+----------+
   | |ex13|   | |ex14|   |
   +----------+----------+



The persistence plot demonstrates accurate detection of significant peaks with proper preprocessing.

.. code:: python

	fp.plot_persistence()

.. image:: ../figs/2dpeaks_perss.png
  :width: 600




.. include:: add_bottom.add