
Mask
''''''''''

The mask method is a local maximum filtering approach for detecting peaks in 2D data (images) using :func:`findpeaks.stats.mask`. This method is particularly effective for identifying prominent local maxima in spatial data.

**Algorithm Overview:**
The mask method employs a multi-step process to identify peaks:

    1. **8-connected neighborhood analysis**: Establishes connectivity patterns for each pixel
    2. **Local maximum filtering**: Identifies pixels with maximal values in their neighborhood
    3. **Background removal**: Eliminates background noise using a threshold-based approach
    4. **Peak isolation**: Creates a final mask containing only the significant peaks

The method uses the input parameter *limit* to define the background threshold, where `background = (X <= limit)`. The background is then eroded to separate peaks from the background. If the limit is set too low, artifacts may appear along the background border due to the local maximum filter's characteristics.

The final mask, containing only the detected peaks, is derived by performing an XOR operation between the local maximum mask and the background.

Two-dimensional data
----------------------------------------------------

The *mask* method is specifically designed for 2D image data and provides excellent results for spatial peak detection using :func:`findpeaks.findpeaks.findpeaks.peaks2d`. Below is a comprehensive example:

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize with mask method
    fp = findpeaks(method='mask')
    # Example 2d image
    X = fp.import_example('2dpeaks')
    # Fit mask method on the 2D data
    results = fp.fit(X)
    # The output contains multiple variables
    print(results.keys())
    # dict_keys(['Xraw', 'Xproc', 'Xdetect'])

The output is a dictionary containing multiple variables that can be used for follow-up analysis.
Details about the input/output parameters can be found here: :func:`findpeaks.stats.mask`
The output variables **Xdetect** and **Xranked** have the same shape as the input data. Elements with values > 0 indicate regions of interest (detected peaks).

Plot the image with the detected peaks using :func:`findpeaks.findpeaks.findpeaks.plot`:

.. code:: python

    # Plot results with horizontal layout
    fp.plot(figure_order='horizontal')

.. _Figure_6:

.. figure:: ../figs/2dpeaks_mask.png




.. include:: add_bottom.add