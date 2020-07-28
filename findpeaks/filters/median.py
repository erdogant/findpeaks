#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2012 - 2013
# Matías Herranz <matiasherranz@gmail.com>
# Joaquín Tita <joaquintita@gmail.com>
#
# https://github.com/PyRadar/pyradar
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


def median_filter(img, win_size=3):
    """
    Apply a 'median filter' to 'img' with a window size equal to 'win_size'.
    Parameters:
        - img: a numpy matrix representing the image.
        - win_size: the size of the windows (by default 3)
    """

    # assert_window_size(win_size)
    if len(img.shape) > 2: raise Exception('ERROR: Image should be 2D.')
    if win_size < 3: raise Exception('ERROR: win size must be at least 3')

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
