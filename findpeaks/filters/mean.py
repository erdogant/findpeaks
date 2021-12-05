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


def mean_filter(img, win_size=3):
    """Mean filter.

    Description
    -----------
    Apply a 'mean filter' to 'img' with a window size equal to 'win_size'.

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
    >>> img_filtered = findpeaks.stats.mean_filter(img.copy(), win_size=15)
    >>>
    >>> plt.figure()
    >>> fig, axs = plt.subplots(1,2)
    >>> axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
    >>> axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Mean filter')

    """
    if win_size < 3: raise Exception('[findpeaks] >ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('[findpeaks] >ERROR: Image should be 2D. Hint: set the parameter: togray=True')
    if ((win_size % 2) == 0): print('[findpeaks] >It is highly recommended to user odd window sizes. You provided %s, an even number.' % (win_size))

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
            window_mean = window.mean()

            if (window_mean is None) or np.isnan(window_mean):
                window_mean = 0

            img_filtered[i, j] = round(window_mean)

    return img_filtered


def assert_indices_in_range(width, height, xleft, xright, yup, ydown):
    """
    Asserts index out of image range.
    """
    assert xleft >= 0 and xleft <= width, \
        "index xleft:%s out of range (%s<= xleft < %s)" % (xleft, 0, width)

    assert xright >= 0 and xright <= width, \
        "index xright:%s out of range (%s<= xright < %s)" % (xright, 0, width)

    assert yup >= 0 and yup <= height, \
        "index yup:%s out of range. (%s<= yup < %s)" % (yup, 0, height)

    assert ydown >= 0 and ydown <= height, \
        "index ydown:%s out of range. (%s<= ydown < %s)" % (ydown, 0, height)
