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


def _weighting(window, cu=0.25):
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

    The Additive Noise Lee Despeckling Filter
    Let's assume that the despeckling noise is additive with a constant mean of zero, a constant variance, and drawn from a Gaussian distribution. Use a window (I x J pixels) to scan the image with a stride of 1 pixels (and I will use reflective boundary conditions). The despeckled value of the pixel in the center of the window located in the ith row and jth column is, zhat_ij = mu_k + W*(z_ij = mu_z), where mu_k is the mean value of all pixels in the window centered on pixel i,j, z_ij is the unfiltered value of the pixel, and W is a weight calculated as, W = var_k / (var_k + var_noise), where var_k is the variance of all pixels in the window and var_noise is the variance of the speckle noise. A possible alternative to using the actual value of the center pixel for z_ij is to use the median pixel value in the window.
    The parameters of the filter are the window/kernel size and the variance of the noise (which is unknown but can perhaps be estimated from the image as the variance over a uniform feature smooth like the surface of still water). Using a larger window size and noise variance will increase radiometric resolution at the expense of spatial resolution.
    For more info on the Lee Filter and other despeckling filters see http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/speckle-function.htm
    Assumes noise mean = 0
    If you don't want the window to be a square of size x size, just replace uniform_filter with something else (convolution with a disk, gaussian filter, etc). Any type of (weighted) averaging filter will do, as long as it is the same for calculating both img_mean and img_square_mean.
    The Lee filter seems rather old-fashioned as a filter. It won't behave well at edges because for any window that has an edge in it, the variance is going to be much higher than the overall image variance, and therefore the weights (of the unfiltered image relative to the filtered image) are going to be close to 1.


    """
    if win_size < 3: raise Exception('[findpeaks] >ERROR: window size (win_size) must be at least 3')
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
            w_t = _weighting(window, cu)
            window_mean = window.mean()

            new_pix_value = (pix_value * w_t) + (window_mean * (1.0 - w_t))

            if new_pix_value < 0.0: raise Exception("ERROR: lee_filter(), pixel filtered can't be negative")
            if (new_pix_value is None) or np.isnan(new_pix_value):
                new_pix_value = 0

            img_filtered[i, j] = round(new_pix_value)

    return img_filtered
