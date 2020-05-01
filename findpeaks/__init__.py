from findpeaks.findpeaks import (
    fit,
	plot,
)

from findpeaks.utils.smoothline import smooth_line1d, smooth_line2d

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.0'

# module level doc-string
__doc__ = """
findpeaks
=====================================================================

Description
-----------
findpeaks is for the detection of peaks and valleys in a 1D vector.

Examples
--------
>>> import findpeaks
>>> X = [10,11,9,23,21,11,45,20,11,12]
>>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
>>> out = findpeaks.fit(X)
>>> findpeaks.plot(out)

References
----------
* https://github.com/erdogant/findpeaks
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

"""
