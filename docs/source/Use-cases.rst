.. _code_directive:

-------------------------------------

*SAR* and *SONAR* images are affected by *speckle* noise that inherently exists in and which degrades the image quality.
It is caused by the back-scatter waves from multiple distributed targets. It is locally strong and it increases the mean Grey level of local area.
Reducing the noise enhances the resolution but tends to decrease the spatial resolution too.


SONAR
''''''''''

Sonar images are corrupted by speckle noise, and peak detection is very dificult or may not even be possible.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Import image example
    img = fp.import_example('2dpeaks_image')
    # Initializatie
    fp = findpeaks(scale=None, denoise=None, togray=True, imsize=(300,300))
    # Fit
    fp.fit(img)
    # Thousands of peaks are detected at this point.
    fp.plot()
    fp.plot_mesh()


.. |figU9| image:: ../figs/sonar_plot_no_preprocessing.png
.. |figU10| image:: ../figs/sonar_mesh_no_preprocessing.png

.. table:: Results without pre-processing
   :align: center

   +----------+
   | |figU9|  |
   +----------+
   | |figU10| |
   +----------+


From this point on, we will *pre-process* the image and apply the *topology* method for peak detection.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Import image example
    img = fp.import_example('2dpeaks_image')
    # Initializatie
    fp = findpeaks(scale=True, denoise='fastnl', window=31, togray=True, imsize=(300,300))
    # Fit
    fp.fit(img)

At this point, the image is pre-processed and the peaks are detected. First we will examine the results by looking at the pre-processing steps.
Below are depicted the four steps of pre-processing. Note that all images are colored in the same manner but the first three look different because RGB colors are used.
The final denoised picture does show clear removal of the speckle noise. But is it good enough to detect the correct peaks?

.. code:: python

    # Plot
    fp.plot_preprocessing()


.. |figU0| image:: ../figs/sonar_pre_processing.png

.. table:: Pre-processing Sonar image
   :align: center

   +----------+
   | |figU0|  |
   +----------+
   

In the next step, we can examine the detected peaks (see below). But these peaks are barely visible on the plot. Nevertheless, we seem to removed many peaks compared to the not-preprocessed image.

.. code:: python

    # Plot
    fp.plot()


.. |figU1| image:: ../figs/sonar_plot.png

.. table:: Detected peaks
   :align: center

   +----------+
   | |figU1|  |
   +----------+

The detection of peaks and pre-processing steps becomes clear when we create a 3D mesh plot.
Below can be seen that the denoising has done a very good job in reducing the speckle noise and keeping the peak of interest.

.. code:: python

    # Plot
    fp.plot_mesh()
    # Rotate to make a top view
    fp.plot_mesh(view=(90,0))


.. |figU3| image:: ../figs/sonar_mesh1.png
.. |figU4| image:: ../figs/sonar_mesh2.png
.. |figU5| image:: ../figs/sonar_mesh3.png
.. |figU6| image:: ../figs/sonar_mesh4.png

.. table:: Mesh plot. Top: 3D mesh. Bottom: top view.
   :align: center

   +----------+----------+
   | |figU3|  | |figU4|  |
   +----------+----------+
   | |figU5|  | |figU6|  |
   +----------+----------+
   
A deep examination can be done with the persistence-homology plot. See below the code how to do this.
Even after denoising, we detect many peaks along the diagonal which are not of interest (see topology section for more information). Only 5 points are potential peaks of interest.
But this information allows to limit the model, and focus only on the peaks that that are off the diagonal.

.. code:: python

    # Plot
    fp.plot_persistence()

    # Plot the top 15 peaks that are detected and examine the scores
    fp.results['persistence'][1:10]

    +----+-----+-----+---------------+---------------+---------+
    |    |   x |   y |   birth_level |   death_level |   score |
    +====+=====+=====+===============+===============+=========+
    |  0 |  64 | 228 |           228 |             0 |     228 |
    +----+-----+-----+---------------+---------------+---------+
    |  1 | 299 | 114 |           114 |             6 |     108 |
    +----+-----+-----+---------------+---------------+---------+
    |  2 |  52 | 166 |           166 |           103 |      63 |
    +----+-----+-----+---------------+---------------+---------+
    |  3 |  61 | 223 |           223 |           167 |      56 |
    +----+-----+-----+---------------+---------------+---------+
    |  4 |  60 | 217 |           217 |           194 |      23 |
    +----+-----+-----+---------------+---------------+---------+
    |  5 | 288 | 113 |           113 |            92 |      21 |
    +----+-----+-----+---------------+---------------+---------+
    |  6 | 200 | 104 |           104 |            87 |      17 |
    +----+-----+-----+---------------+---------------+---------+
    |  7 | 293 | 112 |           112 |            97 |      15 |
    +----+-----+-----+---------------+---------------+---------+
    |  8 | 110 |  93 |            93 |            78 |      15 |
    +----+-----+-----+---------------+---------------+---------+
    |  9 |  45 | 121 |           121 |           107 |      14 |
    +----+-----+-----+---------------+---------------+---------+

    # Take the minimum score for the top peaks off the diagonal.
    limit = fp.results['persistence'][0:5]['score'].min()
    # Initializatie findpeaks again but now with the limit parameter
    fp_new = findpeaks(scale=True, denoise='fastnl', window=31, togray=True, imsize=(300,300), limit=limit)
    # Fit
    fp_new.fit(img)
    # Plot
    fp_new.plot_persistence()


.. |figU7| image:: ../figs/sonar_persitence.png
.. |figU8| image:: ../figs/sonar_persitence_limit.png

.. table:: persistence-homology. Top: no limit. Bottom: with limit
   :align: center

   +----------+
   | |figU7|  |
   +----------+
   | |figU8|  |
   +----------+

The final results show that peak-detection for Sonar images is possible using a emperical approach.
