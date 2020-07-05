from findpeaks.findpeaks import (
    fit,
	plot,
    import_example,
    peaks1d,
    peaks2d,
    plot_mesh,
    plot_peristence,
    plot_mask,
    plot_preprocessing,
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
findpeaks is for the detection and vizualization of peaks and valleys in a 1D-vector and 2D-array.
In case of 2D-array, the image can be processed by resizing, scaling, and denoising.
Peaks are detected using the masking and the topology method.
Results can be plotted for the preprocessing steps, the persistence of peaks, the final masking plot and a mesh 3d-plot.

Examples
--------
>>> import findpeaks
>>> X = [10,11,9,23,21,11,45,20,11,12]
>>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
>>> out = findpeaks.fit(X, lookahead=1)
>>> findpeaks.plot(out)
>>> out = findpeaks.fit(X, lookahead=1, smooth=10)
>>> findpeaks.plot(out)

References
----------
* https://github.com/erdogant/findpeaks
* https://www.sthu.org/code/codesnippets/imagepers.html
* H. Edelsbrunner and J. Harer, Computational Topology. An Introduction, 2010, ISBN 0-8218-4925-5.
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

"""
