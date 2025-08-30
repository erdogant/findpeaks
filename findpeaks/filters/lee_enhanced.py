#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2012 - 2013
# Matías Herranz <matiasherranz@gmail.com>
# Joaquín Tita <joaquintita@gmail.com>
# https://github.com/PyRadar/pyradar
# 
# 2020: Erdogan Taskesen: <erdogant@gmail.com> Converted to python3.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.


from math import exp
import numpy as np
import logging

logger = logging.getLogger(__name__)

K_DEFAULT = 1.0
CU_DEFAULT = 0.523
CMAX_DEFAULT = 1.73

def weighting(pix_value, window, k=K_DEFAULT, cu=CU_DEFAULT, cmax=CMAX_DEFAULT):
    """
    Computes the weighthing function for Lee filter using cu as the noise
    coefficient.
    """

    # cu is the noise variation coefficient

    # ci is the variation coefficient in the window
    window_mean = window.mean()
    window_std = window.std()
    try:
        ci = window_std / window_mean
    except:
        ci = 0

    if ci <= cu:  # use the mean value
        w_t = 1.0
    elif cu < ci < cmax:  # use the filter
        w_t = exp((-k * (ci - cu)) / (cmax - ci))
    elif ci >= cmax:  # preserve the original value
        w_t = 0.0
    else: # use the mean value
        w_t = 1.0

    return w_t


def assert_parameters(win_size, k, cu, cmax):
    """Asserts parameters in range.
    
    Parameters
    ----------
    win_size : int
        Window size parameter.
    k : float
        k parameter in range [0:10].
    cu : float
        cu parameter, must be positive.
    cmax : float
        cmax parameter, must be positive and greater equal than cu.
    """
    if 0 > k > 10: raise Exception("k parameter out of range 0<= k <= 10, submitted %s" %(k))
    if cu < 0: raise Exception( "cu can't be negative")
    if cmax < 0 and cmax < cu: raise Exception("cmax must be positive and greater equal to cu: %s" %(cu))
    if win_size < 3: raise Exception('ERROR: win size must be at least 3')


def lee_enhanced_filter(img, win_size=3, k=K_DEFAULT, cu=CU_DEFAULT, cmax=CMAX_DEFAULT):
    """Lee enhanced filter.

    Enhanced Lee Filter Description
    -------------------------------
    Apply Enhanced Lee filter to a numpy matrix containing the image, with a window of win_size x win_size.

    Parameters
    ----------
    img : array-like
        Input image.
    win_size : int, int (default: 3)
        Window size.
    cu : float (default: 0.25)
        cu factor.
    k : float, (default: 1.0)
        Kvalue.
    cmax : float, (default: 1.73)
        cmax value.

    Returns
    -------
    img_filtered : array-like
        Filtered image.

    Examples
    --------
    >>> import findpeaks
    >>> import matplotlib.pyplot as plt
    >>> img = findpeaks.import_example('2dpeaks_image')
    >>> # Resize
    >>> img = findpeaks.stats.resize(img, size=(300,300))
    >>> # Make grey image
    >>> img = findpeaks.stats.togray(img)
    >>> # Scale between [0-255]
    >>> img = findpeaks.stats.scale(img)
    >>> # Filter
    >>> img_filtered = findpeaks.stats.lee_enhanced_filter(img.copy(), win_size=15)
    >>>
    >>> plt.figure()
    >>> fig, axs = plt.subplots(1,2)
    >>> axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
    >>> axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Lee enhanced filter')

    """
    if win_size < 3: raise Exception('[findpeaks] >ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('[findpeaks] >ERROR: Image should be 2D. Hint: set the parameter: togray=True')
    if ((win_size % 2) == 0): print('[findpeaks] >It is highly recommended to user odd window sizes. You provided %s, an even number.' % (win_size))
    assert_parameters(win_size, k, cu, cmax)

    # we process the entire img as float64 to avoid type overflow error
    img = np.float64(img)
    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = int(win_size / 2)

    for i in np.arange(0, N):
        xleft = i - win_offset
        xright = i + win_offset

        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N

        for j in np.arange(0, M):
            yup = j - win_offset
            ydown = j + win_offset

            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M

            pix_value = img[i, j]
            window = img[xleft:xright, yup:ydown]
            w_t = weighting(pix_value, window, k, cu, cmax)
            window_mean = window.mean()

            new_pix_value = (window_mean * w_t) + (pix_value * (1.0 - w_t))

            if new_pix_value < 0.0: raise Exception("ERROR: lee_filter(), pixel filtered can't be negative")
            if (new_pix_value is None) or np.isnan(new_pix_value):
                new_pix_value = 0

            img_filtered[i, j] = round(new_pix_value)

    return img_filtered
