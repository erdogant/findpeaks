from findpeaks.findpeaks import findpeaks

from findpeaks.utils.interpolate import interpolate_line1d, interpolate_line2d
import findpeaks.utils.stats as stats

# Import the denosing filters
from findpeaks.filters.lee import lee_filter
from findpeaks.filters.lee_enhanced import lee_enhanced_filter
from findpeaks.filters.kuan import kuan_filter
from findpeaks.filters.frost import frost_filter
from findpeaks.filters.median import median_filter
from findpeaks.filters.mean import mean_filter



__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '2.1.6'

# module level doc-string
__doc__ = """
findpeaks
=====================================================================

Description
-----------
findpeaks is for the detection and vizualization of peaks and valleys in a 1D-vector and 2D-array.
In case of 2D-array, the image can be pre-processed by resizing, scaling, and denoising.
Peaks are detected using the masking and the topology method.
The results can be plotted for the preprocessing steps, the persistence of peaks, the final masking plot and a 3d-mesh plot.

Examples
--------
>>> from findpeaks import findpeaks
>>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
>>> fp = findpeaks(interpolate=10, lookahead=1)
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # 2D array example
>>> from findpeaks import findpeaks
>>> fp = findpeaks(method='topology')
>>> X = fp.import_example('2dpeaks')
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # Image example
>>> from findpeaks import findpeaks
>>> fp = findpeaks(method='topology', denoise='fastnl', window=30, imsize=(300,300))
>>> X = fp.import_example('2dpeaks_image')
>>> results = fp.fit(X)
>>> fp.plot()
>>>
>>> # Plot each seperately
>>> fp.plot_preprocessing()
>>> fp.plot_persistence()
>>> fp.plot_mesh()

References
----------
* https://github.com/erdogant/findpeaks
* https://www.sthu.org/code/codesnippets/imagepers.html
* H. Edelsbrunner and J. Harer, Computational Topology. An Introduction, 2010, ISBN 0-8218-4925-5.
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

"""
