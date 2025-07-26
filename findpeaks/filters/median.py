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
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {msg}', style='{', datefmt='%d-%m-%Y %H:%M:%S')


def median_filter(img, win_size=3):
    """Median Filter.
    
    Description
    -----------
    Apply a 'median filter' to 'img' with a window size equal to 'win_size'.


    Parameters
    ----------
    img : array-like
        Input image.
    win_size : int, int (default: 3)
        The size of the windows.

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
    >>> img_filtered = findpeaks.stats.median_filter(img.copy(), win_size=15)
    >>>
    >>> plt.figure()
    >>> fig, axs = plt.subplots(1,2)
    >>> axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
    >>> axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Median filter')

    """
    # assert_window_size(win_size)
    if win_size < 3: raise Exception('ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('ERROR: Image should be 2D. Hint: set the parameter: togray=True')

    N, M = img.shape
    win_offset = int(win_size / 2)
    img_filtered = np.zeros_like(img)

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

            # assert_indices_in_range(N, M, xleft, xright, yup, ydown)

            window = img[xleft:xright, yup:ydown]
            window_median = np.median(window)
            if (window_median is None) or np.isnan(window_median):
                window_median = 0

            img_filtered[i, j] = round(window_median)

    return img_filtered
