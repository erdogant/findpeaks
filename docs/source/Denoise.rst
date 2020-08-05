.. _code_directive:

-------------------------------------

Denoise
''''''''''

The mask method takes an image and detect the peaks using the local maximum filter.
Multiple steps are involved in this approach, first an 8-connected neighborhood is set.
Then, the local maximum filter is applied, and pixel of maximal value in their neighborhood are set to 1.

In order to isolate the peaks we must remove the background from the mask. The background is simply created by the input parameter *limit* so that the background = (X <= limit)
The background is eroded to subtract the peaks from the background. If the limit is to small for example, a line will appear along the background border (artifact of the local maximum filter).

The final mask, containing only peaks, is derived by removing the background from the local_max mask (xor operation).


Lee
----------------------------------------------------

:func:`findpeaks.filters.lee.lee_filter`


Lee enhanced
----------------------------------------------------

:func:`findpeaks.filters.lee_enhanced.lee_enhanced_filter`

Kuan
----------------------------------------------------

:func:`findpeaks.filters.kuan.kuan_filter`


Frost
----------------------------------------------------

:func:`findpeaks.filters.frost.frost_filter`

Mean
----------------------------------------------------

:func:`findpeaks.filters.mean.mean_filter`


Median
----------------------------------------------------

:func:`findpeaks.filters.median.median_filter`


fastnl
----------------------------------------------------

:func:`findpeaks.utils.stats.denoise`


bilateral
----------------------------------------------------

:func:`findpeaks.utils.stats.denoise`

