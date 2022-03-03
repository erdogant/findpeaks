.. _code_directive:

-------------------------------------

Mask
''''''''''

The mask method takes an image and detect the peaks using the local maximum filter.
Multiple steps are involved in this approach, first an 8-connected neighborhood is set.
Then, the local maximum filter is applied, and pixel of maximal value in their neighborhood are set to 1.

In order to isolate the peaks we must remove the background from the mask. The background is simply created by the input parameter *limit* so that the background = (X <= limit)
The background is eroded to subtract the peaks from the background. If the limit is to small for example, a line will appear along the background border (artifact of the local maximum filter).

The final mask, containing only peaks, is derived by removing the background from the local_max mask (xor operation).


Two-dimensional data
----------------------------------------------------

The *mask* method is only avaiable for 2d-image data. Below is shown an example:


.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize
    fp = findpeaks(method='mask')
    # Example 2d image
    X = fp.import_example('2dpeaks')
    # Fit topology method on the 1d-vector
    results = fp.fit(X)
    # The output contains multiple variables
    print(results.keys())
    # dict_keys(['Xraw', 'Xproc', 'Xdetect'])

The output is a dictionary containing multiple variables that can be of use for follow-up analysis.
Details about the input/output parameters can be found here: :func:`findpeaks.stats.mask`
The output variables **Xdetect** and **Xranked** has the same shape as the input data. The elements with value > 0 depict a region of interest.


Plot the image with the detected peaks:

.. code:: python

    # Import library
    fp.plot()

.. _Figure_6:

.. figure:: ../figs/2dpeaks_mask.png


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
