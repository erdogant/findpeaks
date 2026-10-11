
findpeaks
---------

Python library for robust detection and analysis of peaks and valleys in 1D and 2D data.

`findpeaks` is a comprehensive library for detecting, analyzing, and visualizing peaks and valleys 
in both 1D vectors and 2D arrays (images). It provides multiple detection methods including 
topology-based persistent homology, traditional peak detection algorithms, and mask-based approaches.

Key Features
************

* **Multiple Detection Methods**: Topology (persistent homology), peakdetect,
  caerus, and mask-based detection
* **1D and 2D Support**: Works with time series, signals, images, and spatial
  data
* **Advanced Preprocessing**: Denoising, scaling, interpolation, and image
  preprocessing
* **Rich Visualization**: Persistence diagrams, 3D mesh plots, preprocessing
  steps, and masking plots
* **Robust Analysis**: Mathematically stable methods with quantitative
  significance measures

Detection Methods
*****************

Topology
	Uses persistent homology for robust peak detection with mathematical stability
Peakdetect
	Traditional algorithm for 1D peak detection with lookahead and delta parameters
Caerus
	Advanced peak detection with multiple optimization strategies
Mask
	Local maximum filtering for 2D peak detection

**Applications**

* Signal processing and time series analysis
* Image processing and feature detection
* Hough Transform applications with enhanced robustness
* Scientific data analysis and visualization
* Peak finding in noisy or complex datasets

Examples
********

>>> from findpeaks import findpeaks
>>> 
>>> # 1D vector example
>>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
>>> fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10)
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # 2D array example
>>> X = fp.import_example('2dpeaks')
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # Image processing example
>>> fp = findpeaks(method='topology', imsize=(300,300), denoise='fastnl', params={'window': 30})
>>> X = fp.import_example('2dpeaks_image')
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # Advanced visualization
>>> fp.plot_preprocessing()  # Show preprocessing steps
>>> fp.plot_persistence()	 # Show persistence diagram
>>> fp.plot_mesh()			# Show 3D mesh plot

References
**********
* https://erdogant.github.io/findpeaks/
* Johannes Ferner et al, Persistence-based Hough Transform for Line Detection, https://arxiv.org/abs/2504.16114

Components
**********

.. automodule:: findpeaks.findpeaks
   :members:
   :undoc-members:
   :member-order: bysource


.. include:: ../add/add_bottom.add
