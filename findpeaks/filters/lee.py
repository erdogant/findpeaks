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


def weighting(window, cu=0.25):
    """
    Computes the weighthing function for Lee filter using cu as the noise
    coefficient.
    """
    # cu is the noise variation coefficient
    two_cu = cu * cu

    # ci is the variation coefficient in the window
    window_mean = window.mean()
    window_std = window.std()
    ci = window_std / window_mean

    two_ci = ci * ci

    if not two_ci:  # dirty patch to avoid zero division
        two_ci = 0.01

    if cu > ci:  # preserve the original value (I guess)
        w_t = 0.0
    else: # use the filter
        w_t = 1.0 - (two_cu / two_ci)

    return w_t


def lee_filter(img, win_size=3, cu=0.25):
    """
    Apply lee to a numpy matrix containing the image, with a window of
    win_size x win_size.
    """
    if win_size < 3: raise Exception('[findpeaks] >ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('[findpeaks] >ERROR: Image should be 2D. Hint: set the parameter: togray=True')
    if ((win_size % 2) == 0): print('[findpeaks] >It is highly recommended to user odd window sizes. You provided %s, an even number.' % (win_size))

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

            # assert_indices_in_range(N, M, xleft, xright, yup, ydown)

            pix_value = img[i, j]
            window = img[xleft:xright, yup:ydown]
            w_t = weighting(window, cu)
            window_mean = window.mean()

            new_pix_value = (pix_value * w_t) + (window_mean * (1.0 - w_t))

            if new_pix_value < 0.0: raise Exception("ERROR: lee_filter(), pixel filtered can't be negative")
            if (new_pix_value is None) or np.isnan(new_pix_value):
                new_pix_value = 0

            img_filtered[i, j] = round(new_pix_value)

    return img_filtered
