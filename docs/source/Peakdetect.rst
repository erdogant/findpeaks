.. _code_directive:

-------------------------------------

Peakdetect
''''''''''''

The library ``peakdetect`` [1] is based on Billauers work [2] and this gist [3]. The method is directly incorporated in ``findpeaks`` and has a strong advantage to
find the local maxima and minima in noisy signal. Noisy data is very common in real-life signals, which makes methods such as zero-derivates not applicable.
The typical solution is to smooth the curve with some low-pass filter but this comes with the trade-off that the peaks in the original signal may be lost.
This method works only for one-dimensional data.

One-dimensional data
----------------------------------------------------

For the **peakdetect** method, we need to set the **lookahead** parameter, which is the distance to look ahead from a peak candidate to determine if it is the actual peak.
The default value is set to 200 but this value is way too large for small datasets (i.e., with <50 data points).

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize
    fp = findpeaks(method='peakdetect', lookahead=1, interpolate=None)
    # Example 1d-vector
    X = fp.import_example('1dpeaks')
    # Fit topology method on the 1d-vector
    results = fp.fit(X)
    # The output contains multiple variables
    print(results.keys())
    # dict_keys([df'])

    +----+-----+------+--------+----------+--------+
    |    |   x |    y |   labx | valley   | peak   |
    +====+=====+======+========+==========+========+
    |  0 |   0 | 1.5  |      1 | True     | False  |
    +----+-----+------+--------+----------+--------+
    |  1 |   1 | 0.8  |      1 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  2 |   2 | 1.2  |      1 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  3 |   3 | 0.2  |      2 | True     | False  |
    +----+-----+------+--------+----------+--------+
    |  4 |   4 | 0.4  |      2 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  5 |   5 | 0.39 |      2 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  6 |   6 | 0.42 |      2 | False    | True   |
    +----+-----+------+--------+----------+--------+
    |  7 |   7 | 0.22 |      2 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  8 |   8 | 0.23 |      2 | False    | False  |
    +----+-----+------+--------+----------+--------+
    |  9 |   9 | 0.1  |      3 | True     | False  |
    +----+-----+------+--------+----------+--------+
    | 10 |  10 | 0.11 |      3 | False    | False  |
    +----+-----+------+--------+----------+--------+
    | 11 |  11 | 0.1  |      3 | False    | False  |
    +----+-----+------+--------+----------+--------+
    | 12 |  12 | 0.14 |      3 | False    | True   |
    +----+-----+------+--------+----------+--------+
    | 13 |  13 | 0.09 |      3 | False    | False  |
    +----+-----+------+--------+----------+--------+
    | 14 |  14 | 0.04 |      3 | False    | False  |
    +----+-----+------+--------+----------+--------+
    | 15 |  15 | 0.02 |      3 | False    | False  |
    +----+-----+------+--------+----------+--------+
    | 16 |  16 | 0.01 |      3 | True     | False  |
    +----+-----+------+--------+----------+--------+

The output is a dictionary containing a single dataframe (*df*) that can be of use for follow-up analysis. See: :func:`findpeaks.findpeaks.findpeaks.peaks1d`

.. _Figure_7:

.. figure:: ../figs/1dpeaks_peakdetect.png


The strength of this approach becomes visible when we use a noisy dataset.

.. code:: python

    # Import library
    from findpeaks import findpeaks
    # Initialize
    fp = findpeaks(method='peakdetect', lookahead=200, interpolate=None)

    # Example 1d-vector
    i = 10000
    xs = np.linspace(0,3.7*np.pi,i)
    X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

    # Fit topology method on the 1d-vector
    results = fp.fit(X)
    # Plot
    fp.plot()

.. _Figure_8:

.. figure:: ../figs/fig3.png


References peakdetect
-----------------------
    * [1] https://github.com/anaxilaus/peakdetect
    * [2] http://billauer.co.il/peakdet.html
    * [3] https://gist.github.com/sixtenbe/1178136
