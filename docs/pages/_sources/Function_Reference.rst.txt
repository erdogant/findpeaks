Function Reference
==================

This page provides a comprehensive reference of all functions available in the findpeaks library, organized by category for easy navigation.

Core Functions
---------------

.. list-table:: Main findpeaks class functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.findpeaks.findpeaks`
     - Main class for peak detection and analysis
   * - :func:`findpeaks.findpeaks.findpeaks.__init__`
     - Initialize the findpeaks object with detection parameters
   * - :func:`findpeaks.findpeaks.findpeaks.fit`
     - Fit the peak detection model to the input data
   * - :func:`findpeaks.findpeaks.findpeaks.peaks1d`
     - Detect peaks and valleys in 1D data
   * - :func:`findpeaks.findpeaks.findpeaks.peaks2d`
     - Detect peaks and valleys in 2D data (images)
   * - :func:`findpeaks.findpeaks.findpeaks.preprocessing`
     - Apply preprocessing pipeline to the input data
   * - :func:`findpeaks.findpeaks.findpeaks.import_example`
     - Import example datasets for testing and demonstration

Visualization Functions
------------------------

.. list-table:: Plotting and visualization functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.findpeaks.findpeaks.plot`
     - Main plotting function for comprehensive results visualization
   * - :func:`findpeaks.findpeaks.findpeaks.plot1d`
     - Plot results for 1D data analysis
   * - :func:`findpeaks.findpeaks.findpeaks.plot2d`
     - Plot results for 2D data analysis
   * - :func:`findpeaks.findpeaks.findpeaks.plot_persistence`
     - Plot persistence diagrams for topology method
   * - :func:`findpeaks.findpeaks.findpeaks.plot_mesh`
     - Create 3D mesh visualizations
   * - :func:`findpeaks.findpeaks.findpeaks.plot_preprocessing`
     - Visualize preprocessing pipeline steps
   * - :func:`findpeaks.findpeaks.findpeaks.plot_mask`
     - Plot masking results with customizable font size

Peak Detection Methods
-----------------------

.. list-table:: Core detection algorithms
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.stats.topology`
     - Topology-based peak detection using persistent homology
   * - :func:`findpeaks.stats.topology2d`
     - 2D topology-based peak detection
   * - :func:`findpeaks.peakdetect.peakdetect`
     - Traditional peak detection algorithm
   * - :func:`findpeaks.stats.mask`
     - Mask-based local maximum filtering for 2D data
   * - :func:`findpeaks.stats.caerus`
     - Caerus method for financial time series analysis

Preprocessing Functions
-------------------------

.. list-table:: Data preprocessing and transformation functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.interpolate.interpolate_line1d`
     - 1D data interpolation with multiple methods
   * - :func:`findpeaks.interpolate.interpolate_line2d`
     - 2D data interpolation for x,y coordinates
   * - :func:`findpeaks.stats.resize`
     - Resize 2D arrays (images) using OpenCV
   * - :func:`findpeaks.stats.scale`
     - Scale data to specified range
   * - :func:`findpeaks.stats.togray`
     - Convert images to grayscale
   * - :func:`findpeaks.stats.denoise`
     - Apply various denoising filters to images

Denoising Filters
------------------

.. list-table:: Image denoising filter functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.filters.lee.lee_filter`
     - Lee filter for additive noise reduction
   * - :func:`findpeaks.filters.lee_enhanced.lee_enhanced_filter`
     - Enhanced Lee filter with improved edge preservation
   * - :func:`findpeaks.filters.lee_sigma.lee_sigma_filter`
     - Lee Sigma filter for adaptive noise reduction
   * - :func:`findpeaks.filters.frost.frost_filter`
     - Frost filter for multiplicative noise reduction
   * - :func:`findpeaks.filters.kuan.kuan_filter`
     - Kuan filter for speckle noise reduction
   * - :func:`findpeaks.filters.mean.mean_filter`
     - Mean filter for noise smoothing
   * - :func:`findpeaks.filters.median.median_filter`
     - Median filter for noise reduction while preserving edges

Utility Functions
-------------------

.. list-table:: Utility and helper functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`findpeaks.stats._make_unique`
     - Make array elements unique with progress tracking
   * - :func:`findpeaks.union_find.UnionFind`
     - Union-find data structure for connected components
   * - :func:`findpeaks.filters.lee_enhanced.assert_parameters`
     - Validate parameters for enhanced Lee filter

Function Categories by Use Case
--------------------------------

.. list-table:: Functions organized by common use cases
   :widths: 30 70
   :header-rows: 1

   * - Use Case
     - Relevant Functions
   * - **1D Signal Analysis**
     - :func:`findpeaks.findpeaks.findpeaks.peaks1d`, :func:`findpeaks.stats.topology`, :func:`findpeaks.peakdetect.peakdetect`, :func:`findpeaks.interpolate.interpolate_line1d`, :func:`findpeaks.findpeaks.findpeaks.plot1d`
   * - **2D Image Analysis**
     - :func:`findpeaks.findpeaks.findpeaks.peaks2d`, :func:`findpeaks.stats.mask`, :func:`findpeaks.stats.topology2d`, :func:`findpeaks.stats.resize`, :func:`findpeaks.stats.togray`, :func:`findpeaks.findpeaks.findpeaks.plot2d`
   * - **Financial Time Series**
     - :func:`findpeaks.stats.caerus`, :func:`findpeaks.findpeaks.findpeaks.fit`, :func:`findpeaks.findpeaks.findpeaks.plot`
   * - **SAR Image Processing**
     - :func:`findpeaks.stats.denoise`, :func:`findpeaks.filters.lee.lee_filter`, :func:`findpeaks.filters.frost.frost_filter`, :func:`findpeaks.filters.kuan.kuan_filter`, :func:`findpeaks.stats.scale`
   * - **Data Visualization**
     - :func:`findpeaks.findpeaks.findpeaks.plot_persistence`, :func:`findpeaks.findpeaks.findpeaks.plot_mesh`, :func:`findpeaks.findpeaks.findpeaks.plot_preprocessing`, :func:`findpeaks.findpeaks.findpeaks.plot_mask`
   * - **Data Preprocessing**
     - :func:`findpeaks.findpeaks.findpeaks.preprocessing`, :func:`findpeaks.interpolate.interpolate_line1d`, :func:`findpeaks.interpolate.interpolate_line2d`, :func:`findpeaks.stats.resize`, :func:`findpeaks.stats.scale`

Quick Reference by Method
---------------------------

.. list-table:: Functions grouped by detection method
   :widths: 25 75
   :header-rows: 1

   * - Method
     - Functions
   * - **Topology**
     - :func:`findpeaks.stats.topology`, :func:`findpeaks.stats.topology2d`, :func:`findpeaks.findpeaks.findpeaks.plot_persistence`
   * - **Peakdetect**
     - :func:`findpeaks.peakdetect.peakdetect`, :func:`findpeaks.findpeaks.findpeaks.peaks1d`
   * - **Mask**
     - :func:`findpeaks.stats.mask`, :func:`findpeaks.findpeaks.findpeaks.peaks2d`, :func:`findpeaks.findpeaks.findpeaks.plot_mask`
   * - **Caerus**
     - :func:`findpeaks.stats.caerus`, :func:`findpeaks.findpeaks.findpeaks.fit`
   * - **Interpolation**
     - :func:`findpeaks.interpolate.interpolate_line1d`, :func:`findpeaks.interpolate.interpolate_line2d`
   * - **Denoising**
     - :func:`findpeaks.stats.denoise`, :func:`findpeaks.filters.lee.lee_filter`, :func:`findpeaks.filters.frost.frost_filter`, :func:`findpeaks.filters.kuan.kuan_filter`, :func:`findpeaks.filters.mean.mean_filter`, :func:`findpeaks.filters.median.median_filter`

.. include:: add_bottom.add 