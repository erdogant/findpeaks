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


import numpy as np
from scipy.stats import variation
import logging

logger = logging.getLogger(__name__)

def compute_coef_var(image, x_start, x_end, y_start, y_end):
    """
    Compute coefficient of variation in a window of [x_start: x_end] and
    [y_start:y_end] within the image.
    """
    if x_start < 0: raise Exception('ERROR: x_start must be >= 0.')
    if y_start < 0: raise Exception('ERROR: y_start must be >= 0.')

    x_size, y_size = image.shape
    x_overflow = x_end > x_size
    y_overflow = y_end > y_size

    if x_overflow: raise Exception('ERROR: invalid parameters cause x window overflow.')
    if y_overflow: raise Exception('ERROR: invalid parameters cause y window overflow.')

    window = image[x_start:x_end, y_start:y_end]

    coef_var = variation(window, None)

    if not coef_var:  # dirty patch
        coef_var = 0.01
        # print "squared_coef was equal zero but replaced by %s" % coef_var
    if coef_var <= 0: raise Exception('ERROR: coeffient of variation cannot be zero.')

    return coef_var


def calculate_all_Mi(window_flat, factor_A, window):
    """
    Compute all the weights of pixels in the window.
    """
    N, M = window.shape
    center_pixel = np.float64(window[int(N/2), int(M/2)])
    window_flat = window_flat.astype(np.float64)

    distances = np.abs(window_flat - center_pixel)

    weights = np.exp(-factor_A * distances)

    return weights


def calculate_local_weight_matrix(window, factor_A):
    """
    Returns an array with the weights for the pixels in the given window.
    """
    weights_array = np.zeros(window.size)
    window_flat = window.flatten()

    weights_array = calculate_all_Mi(window_flat, factor_A, window)

    return weights_array


def frost_filter(img, damping_factor=2.0, win_size=3):
    """Frost filter.
    
    Frost Filter Description
    ------------------------
    Apply frost filter to a numpy matrix containing the image, with a window of win_size x win_size. By default, the window size is 3x3.

    Parameters
    ----------
    img : array-like
        Input image.
    damping_factor : float (default: 2.0)
        Damping factor.
    win_size : int, int (default: 3)
        Window size.

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
    >>> # frost filter
    >>> img_filtered = findpeaks.stats.frost_filter(img.copy(), damping_factor=2, win_size=15)
    >>>
    >>> plt.figure()
    >>> fig, axs = plt.subplots(1,2)
    >>> axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
    >>> axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Frost filter')

    """
    if win_size < 3: raise Exception('[findpeaks] >ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('[findpeaks] >ERROR: Image should be 2D. Hint: set the parameter: togray=True')
    if ((win_size % 2) == 0): print('[findpeaks] >It is highly recommended to user odd window sizes. You provided %s, an even number.' % (win_size))

    img_filtered = np.zeros_like(img)
    N, M = img.shape
    win_offset = int(win_size / 2)

    for i in np.arange(0, N):
        xleft = i - win_offset
        xright = i + win_offset
        if xleft < 0:
            xleft = 0
        if xright >= N:
            xright = N - 1
        for j in np.arange(0, M):
            yup = j - win_offset
            ydown = j + win_offset
            if yup < 0:
                yup = 0
            if ydown >= M:
                ydown = M - 1

            # inspired by http://www.pcigeomatics.com/cgi-bin/pcihlp/FFROST
            variation_coef = compute_coef_var(img, xleft, xright, yup, ydown)
            window = img[xleft:xright, yup:ydown]
            window_mean = window.mean()
            sigma_zero = variation_coef / window_mean  # var / u^2
            factor_A = damping_factor * sigma_zero

            weights_array = calculate_local_weight_matrix(window, factor_A)
            pixels_array = window.flatten()

            weighted_values = weights_array * pixels_array
            
            new_pix_value  = weighted_values.sum() / weights_array.sum()
            
            if (new_pix_value is None) or np.isnan(new_pix_value):
                new_pix_value = 0

            img_filtered[i, j] = new_pix_value

    return img_filtered
