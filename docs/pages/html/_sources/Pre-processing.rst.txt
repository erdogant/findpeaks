Interpolate/impute
'''''''''''''''''''''

The input parameter "interpolate" extens the data by this factor and is usefull to "smooth" the signal by a (linear) interpolation. It can also handle missing (nan) data!
A smoothed signal can be more robust agains noise, and perform better in the detection of peaks and valleys.
This step can be seen as pre-processing step before applying any method.
The input is 1D numpy vector that can be interpolated by various methods for which the default is **linear**. Note that the initialization of ``findpeaks`` is fixed to **linear**.
If another method is desired, it can be done by directly using the functionality: :func:`findpeaks.interpolate.interpolate_line1d`.

Besides the 1d functionality, there is also a 2d functionlity in case you have x and y-cooridinates: :func:`findpeaks.interpolate.interpolate_line2d`.

    Interpolation methods:
        * String or integer
        * 0 : order degree
        * 1 : order degree
        * 2 : order degree
        * 3 : order degree
        * 'linear'
        * 'nearest'
        * 'zero'
        * 'slinear'
        * 'quadratic'
        * 'cubic'
        * 'previous'
        * 'next'

.. code:: python

    # Import library
    import findpeaks
    # Small dataset
    X = [10,11,9,23,21,11,45,20,11,12]
    # Interpolate the data using linear by factor 10
    Xi = findpeaks.interpolate_line1d(X, method='linear', n=10, showfig=True)
    # Print message
    print('Input data lenth: %s, interpolated length: %s' %(len(X), len(Xi)))
    # Input data lenth: 10, interpolated length: 100


.. |figP0| image:: ../figs/interpolate_example.png

.. table:: Interpolation example
   :align: center

   +----------+
   | |figP0|  |
   +----------+

As mentioned before, the interpolate function :func:`findpeaks.interpolate.interpolate_line1d` can also handle missing data.
Lets demonstrate this by example:

.. code:: python

    # Import library
    import findpeaks
    # Small dataset
    X = [1,2,3,np.nan,np.nan,6,7,np.nan,9,10]
    # Interpolate the data using linear method and n=1. This would not extend the data but simply impute missing values.
    Xi = findpeaks.interpolate_line1d(X, method='linear', n=1)
    print(Xi)
    # array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
   

The interpolate functionality is integrated in ``findpeaks`` by specifying the **interpolate** as the factor *n*.
The advantage of the interpolation integration in findpeaks is the automatic mapping of the results back to the original data and *imputing* missing data.
Otherwise, the detected peaks coordinates on the x-axis would always be different then for the input-data as the data is extended by interpolation.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Init
    fp = findpeaks(method='peakdetect', interpolate=10, lookahead=1)
    # Small dataset
    X = [10,11,9,23,21,11,45,20,11,12]
    # Interpolate the data using linear by factor 10
    results = fp.fit(X)
    fp.plot()
        

.. |figP1| image:: ../figs/fig2_peakdetect.png

.. |figP2| image:: ../figs/fig2_peakdetect_int.png


.. table:: Results without interpolation (left) and with (right)
   :align: center

   +----------+----------+
   | |figP1|  | |figP2|  |
   +----------+----------+
   

Resize
''''''''''''

The resize function :func:`findpeaks.stats.resize` is only applicable for 2D-arrays (images).
The function resizes the images using functionality of ``python-opencv`` using default parameter settings.


Scale
''''''''''''

The *scale* function :func:`findpeaks.stats.scale` is only applicable for 2D-arrays (images).
Scaling data is an import pre-processing step to make sure all data is ranged between the minimum and maximum range.

The images are scaled between [0-255] by the following equation:

    Ximg * (255 / max(Ximg))


Gray
''''''''''''

The *gray* function :func:`findpeaks.stats.togray` is only applicable for 2D-arrays (images).
The function sets the color to gray using functionality of ``python-opencv`` using the ``cv2.COLOR_BGR2GRAY`` settings.


Preprocessing
''''''''''''''

The preprocessing function is developed to pipeline the above mentioned functionalities :func:`findpeaks.findpeaks.findpeaks.preprocessing`.

The pre-processing has 4 (optional) steps and are exectued in this order. After the last step, the peak detection method is applied.
    * 1. Resizing (to reduce computation time).
    * 2. Scaling color pixels between [0-255].
    * 3. Conversion to gray-scale.
    * 4. Denoising of the image.

Each of these steps can be controlled by setting the input parameters.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Init
    fp = findpeaks(method="topology", whitelist=['peak'], imsize=(50,100), scale=True, togray=True, denoise=None)
    # Small dataset
    X = fp.import_example("2dpeaks")

    # Interpolate the data using linear by factor 10
    results = fp.fit(X)
    fp.plot(figure_order='horizontal')
    # fp.plot_persistence()


.. |figP3| image:: ../figs/2dpeaks_interpolate.png

.. table:: Interpolation example 2d-array (image)
   :align: center

   +----------+
   | |figP3|  |
   +----------+
   



.. include:: add_bottom.add