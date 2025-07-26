#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2023: Caroline Goehner: <carolinesophie.goehner@eurac.edu> <carosophie.goehner@gmail.com>
# This part of the library was written at Eurac Research in the 
# framework of the project ScaleAgData (SCALING ΑGRICULTURAL SENSOR 
# DATA for an improved monitoring of agri-environmental conditions. 
# Duration: 01/01/2023 - 31/12/2026) funded by the Horizon Europe 
# program under the grant agreement no 101086355.
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
import xarray as xr
from joblib import Parallel, delayed
from joblib import parallel_backend
import logging

logger = logging.getLogger(__name__)


sigma_DEFAULT = 0.9 # for general applications
win_size_DEFAULT = 7
num_looks_DEFAULT = 1
tk_DEFAULT = 5 # as in S1TBX
num_cores_DEFAULT = -1 
data_measure_DEFAULT = "intensity"

def assert_parameters(sigma, win_size, num_looks, tk, data_measure):
    """
    Asserts parameters in range.
    Parameters:
        - sigma: in [0.5, 0.6, 0.7, 0.8, 0.9]
        - win_size: should be odd, at least 3
        - num_looks: in [1, 2, 3, 4]
        - tk: in [5, 6, 7]
        - data_measure: "intensity" or "amplitude"
    """

    if sigma not in [0.5, 0.6, 0.7, 0.8, 0.9]: raise Exception("Sigma parameter has to be 0.5, 0.6, 0.7, 0.8, or 0.9, submitted %s" %(sigma))
    if win_size < 3: raise Exception('ERROR: win size must be at least 3')
    if num_looks not in [1, 2, 3, 4]: raise Exception("num_looks parameter has to be 1, 2, 3 or 4, submitted %s" %(num_looks))
    if tk not in [5, 6, 7]: print('[findpeaks] >For general applications it is recommended to use threshold tk between 5 and 7. You provided %s.' %(tk))
    if data_measure not in ["intensity", "amplitude"]: raise Exception('ERROR: data_measure has to be "intensity" or "amplitude". You provided %s.' %(data_measure))


def ptTar(x, y, img, Z98, tk):
    """
    Detect if the pixel is part of a point target of surrounding pixels
    Parameters:
        - x: int
            X-coordinate of the pixel.
        - y: int
            Y-coordinate of the pixel.
        - img: xarray
            Input image.
        - Z98: ndarray
            Threshold of the 98th percentile of the img.
        - tk: int
            Threshold for number of K neighbouring pixels > Z98 to classify the pixel as point target, typically 5.
    """

    for c in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
        a = x+c[0]
        b = y+c[1]
        win = img[a-1:a+1, b-1:b+1] # 3x3 windows for pixels surrounding the center pixel
        K_win = np.count_nonzero(win >= Z98) # number of pixels outside the Z98
        if K_win >= tk: # is point target
            ptTarget = True
            break
        else: 
            ptTarget = False
            continue
    return(ptTarget)


def lee_sigma_filter(img,
                     sigma = sigma_DEFAULT,
                     win_size = win_size_DEFAULT,
                     num_looks = num_looks_DEFAULT,
                     tk = tk_DEFAULT,
                     num_cores = num_cores_DEFAULT,
                     data_measure = data_measure_DEFAULT): 
    """Lee sigma filter.

    Description
    -----------
    Improved Lee Sigma, according to Lee Sigma filter in SNAP Sentinel-1 Toolbox.
    Apply the filter with a window of win_size x win_size to a numpy matrix (containing the image), before converting to dB.
    Jong-Sen Lee, Jen-Hung Wen, T. L. Ainsworth, Kun-Shan Chen and A. J. Chen, "Improved Sigma Filter for Speckle Filtering of SAR Imagery",
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 1, pp. 202-213, Jan. 2009, doi: 10.1109/TGRS.2008.2002881.

    Parameters
    ----------
    img : numpy.ndarray or xarray.DataArray
        Input image.
    sigma : float, (default: 0.9)
        Speckle noise standard deviation.
    win_size : int, int (default: 7)
        Window size.
    num_looks : int, (default: 1)
        Number of looks of the SAR img.
    tk: int, (default: 5)
        Threshold of neighbouring pixels outside of the 98th percentile, typically between 5 and 7.
    num_cores: int, (default: -1)
        Number of cores to use for parallel computing, if -1 all CPUs are used, if 1 no parallel computing is used.

    Returns
    -------
    img_filtered : numpy.ndarray or xarray.DataArray
        Filtered image, type depending on input type.

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
    >>> img_filtered = findpeaks.stats.lee_sigma_filter(img.copy(), win_size=7)
    >>>
    >>> plt.figure()
    >>> fig, axs = plt.subplots(1,2)
    >>> axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
    >>> axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Lee sigma filter')

    """

    if win_size < 3: raise Exception('[findpeaks] >ERROR: win size must be at least 3')
    if len(img.shape) > 2: raise Exception('[findpeaks] >ERROR: Image should be 2D. Hint: set the parameter: togray=True')
    if ((win_size % 2) == 0): print('[findpeaks] >It is highly recommended to use odd window sizes. You provided %s, an even number.' % (win_size))
    assert_parameters(sigma, win_size, num_looks, tk, data_measure) # check validity of input parameters

    if data_measure == "intensity":
        if num_looks == 1:
            if sigma == 0.5:
                I1 = 0.436 # lower sigma range
                I2 = 1.920 # upper sigma range
                IsigmaVP = 0.4057 # speckle noise standard deviation (adjusted)

            elif sigma == 0.6:
                I1 = 0.343
                I2 = 2.210
                IsigmaVP = 0.4954
            elif sigma == 0.7:
                I1 = 0.254
                I2 = 2.582
                IsigmaVP = 0.5911
            elif sigma == 0.8:
                I1 = 0.168
                I2 = 3.094
                IsigmaVP = 0.6966
            elif sigma == 0.9:
                I1 = 0.084
                I2 = 3.941
                IsigmaVP = 0.8191

        elif num_looks == 2:
            if sigma == 0.5:
                I1 = 0.582
                I2 = 1.584
                IsigmaVP = 0.2763
            elif sigma == 0.6:
                I1 = 0.501
                I2 = 1.755
                IsigmaVP = 0.3388
            elif sigma == 0.7:
                I1 = 0.418
                I2 = 1.972
                IsigmaVP = 0.4062
            elif sigma == 0.8:
                I1 = 0.327
                I2 = 2.260
                IsigmaVP = 0.4810
            elif sigma == 0.9:
                I1 = 0.221
                I2 = 2.744
                IsigmaVP = 0.5699

        elif num_looks == 3:
            if sigma == 0.5:
                I1 = 0.652
                I2 = 1.458
                IsigmaVP = 0.2222
            elif sigma == 0.6:
                I1 = 0.580
                I2 = 1.586
                IsigmaVP = 0.2736
            elif sigma == 0.7:
                I1 = 0.505
                I2 = 1.751
                IsigmaVP = 0.3280
            elif sigma == 0.8:
                I1 = 0.419
                I2 = 1.965
                IsigmaVP = 0.3892
            elif sigma == 0.9:
                I1 = 0.313
                I2 = 2.320
                IsigmaVP = 0.4624

        elif num_looks == 4:
            if sigma == 0.5:
                I1 = 0.694
                I2 = 1.385
                IsigmaVP = 0.1921
            elif sigma == 0.6:
                I1 = 0.630
                I2 = 1.495
                IsigmaVP = 0.2348
            elif sigma == 0.7:
                I1 = 0.560
                I2 = 1.627
                IsigmaVP = 0.2825
            elif sigma == 0.8:
                I1 = 0.480
                I2 = 1.804
                IsigmaVP = 0.3354
            elif sigma == 0.9:
                I1 = 0.378
                I2 = 2.094
                IsigmaVP = 0.3991
    
    elif data_measure == "amplitude":
        if num_looks == 1:
            if sigma == 0.5:
                A1 = 0.653997
                A2 = 1.40002
                AsigmaVP = 0.208349
            elif sigma == 0.6:
                A1 = 0.578998
                A2 = 1.50601
                AsigmaVP = 0.255358
            elif sigma == 0.7:
                A1 = 0.496999
                A2 = 1.63201
                AsigmaVP = 0.305303
            elif sigma == 0.8:
                A1 = 0.403999
                A2 = 1.79501
                AsigmaVP = 0.361078
            elif sigma == 0.9:
                A1 = 0.286
                A2 = 2.04301
                AsigmaVP = 0.426375
        
        elif num_looks == 2:
            if sigma == 0.5:
                A1 = 0.76
                A2 = 1.263
                AsigmaVP = 0.139021
            elif sigma == 0.6:
                A1 = 0.705
                A2 = 1.332
                AsigmaVP = 0.169777
            elif sigma == 0.7:
                A1 = 0.643
                A2 = 1.412
                AsigmaVP = 0.206675
            elif sigma == 0.8:
                A1 = 0.568
                A2 = 1.515
                AsigmaVP = 0.244576
            elif sigma == 0.9:
                A1 = 0.467
                A2 = 1.673
                AsigmaVP = 0.29107
        
        elif num_looks == 3:
            if sigma == 0.5:
                A1 = 0.806
                A2 = 1.21
                AsigmaVP = 0.109732
            elif sigma == 0.6:
                A1 = 0.76
                A2 = 1.263
                AsigmaVP = 0.138001
            elif sigma == 0.7:
                A1 = 0.708
                A2 = 1.327
                AsigmaVP = 0.163686
            elif sigma == 0.8:
                A1 = 0.645
                A2 = 1.408
                AsigmaVP = 0.19597
            elif sigma == 0.9:
                A1 = 0.557
                A2 = 1.531
                AsigmaVP = 0.234219
        
        elif num_looks == 4:
            if sigma == 0.5:
                A1 = 0.832
                A2 = 1.179
                AsigmaVP = 0.0894192
            elif sigma == 0.6:
                A1 = 0.793
                A2 = 1.226
                AsigmaVP = 0.112018
            elif sigma == 0.7:
                A1 = 0.747
                A2 = 1.279
                AsigmaVP = 0.139243
            elif sigma == 0.8:
                A1 = 0.691
                A2 = 1.347
                AsigmaVP = 0.167771
            elif sigma == 0.9:
                A1 = 0.613
                A2 = 1.452
                AsigmaVP = 0.201036

    # variables
    final_img = None
    if isinstance(img, xr.DataArray): # make it possible to use xarray dataarrays as well 
        final_img = img.copy()
        img = img.values
    win_size_h = int(win_size/2) # "half" window as distance from center pixel in each direction
    
    if data_measure == "intensity":
        sigmaV = 1.0 / (num_looks ** 0.5) # standard deviation of the multiplicative speckle noise, depending on number of looks
        sigmaVP = IsigmaVP
        sigmaRangeLow = I1
        sigmaRangeHigh = I2
    elif data_measure == "amplitude":
        sigmaV = 0.5227 / (num_looks ** 0.5)
        sigmaVP = AsigmaVP
        sigmaRangeLow = A1
        sigmaRangeHigh = A2
    sigmaVSqr = sigmaV**2 # variance of the multiplicative speckle noise
    
    Z98 = np.percentile(img, 98) # threshold of the 98th percentile of the SAR img
    N, M = img.shape
    img_filtered = np.zeros_like(img, dtype=float)  

    def filter_pixel(i, j):
        xleft = i - win_size_h # define left x coordinate of the selected window size
        xright = i + win_size_h+1 # define right x coordinate of the selected window size, add 1 for indexing ndarrays
        if xleft < 0: xleft = 0 # if outside the image dimensions set to min x coordinate
        if xright >= N: xright = N # if outside the image dimensions set to max x coordinate
        
        xleft3 = i - 1 # for 3x3 window
        xright3 = i + 2
        if xleft3 < 0: xleft3 = 0
        if xright3 >= N: xright3 = N
        
        yup = j - win_size_h # in y dimension
        ydown = j + win_size_h+1 
        if yup < 0: yup = 0
        if ydown >= M: ydown = M

        yup3 = j - 1 # for 3x3 window
        ydown3 = j + 2
        if yup3 < 0: yup3 = 0
        if ydown3 >= M: ydown3 = M

        # 1. Point target detection + preservation 
        z = img[i, j] # center pixel value of window
        window = img[xleft:xright, yup:ydown] # window of selected size
        window_3x3 = img[xleft3:xright3, yup3:ydown3]  # 3x3 window

        K = np.count_nonzero(window_3x3 >= Z98) # number of pixels in the 3x3 window outside the Z98

        if (ptTar(i, j, img, Z98, tk) == False # not part of a (earlier) point target
            and (z.item() >= Z98) == False # pixel value is within the 98th percentile of the SAR img
            or ((z.item() >= Z98) == True and (K >= tk) == False) # is not in the 98th percentile, but has enough surrounding pixels that are neither -> it will be filtered
           ): 

            # 2. Pixels selection based on the sigma range
            # - MMSE on 3x3 using orig_sigmaVP to compute a priori mean (priori_x)                   
            mean_z = window_3x3.mean() # local mean in 3x3
            Var_z = window_3x3.var(dtype = np.float64) # local variance in 3x3
            Var_x = (Var_z - mean_z**2 * sigmaVSqr) / (1 + sigmaVSqr) # Variance of x
            if Var_x < 0: Var_x = 0.0 # according to s1tbx
            b = Var_x / (Var_z+1e-50) # weight function - add small values to avoid nan weights when all the values are similar in the window

            priori_x = (1-b) * mean_z + b * z  # MMSE filter to calculate a priori mean

            # - establish sigma range using LUT for sigma in Intensity img and num_looks:
            XsigmaRangeLow = sigmaRangeLow * priori_x # lower sigma range
            XsigmaRangeHigh = sigmaRangeHigh * priori_x # upper sigma range
            sigmaVPSqr = sigmaVP**2 # speckle noise variance

            # - select pixels in window if their values fall into sigma range, compute mean_z and Var_z
            window = window[np.where(np.logical_and(window >= XsigmaRangeLow, window <= XsigmaRangeHigh))]
            if np.count_nonzero(window) == 0: new_pix_value = z # when window is empty, according to S1TBX
            else:
                mean_z = window.mean() # local mean in the sigma range
                Var_z = window.var(dtype = np.float64) # local variance in the sigma range

                # 3. MMSE application
                # - compute MMSE filter weight b using Var_x, based on mean_z, Var_z and sigmaVPSqr
                Var_x = (Var_z - mean_z**2 * sigmaVPSqr) / (1 + sigmaVPSqr) # Variance of x
                if Var_x < 0: Var_x = 0.0 # according to s1tbx
                b = Var_x / (Var_z+1e-50) # weight function - add small values to avoid nan weights when all the values are similar in the window

                # - filter center pixel using MMSE
                new_pix_value = (1-b) * mean_z + b * z # new filtered pixel value
            

        else: # center pixel is part of a (earlier) point target or is a point target pixel -> it will NOT be filtered
            new_pix_value = z
            
        return new_pix_value
    
    # Parallel Process
    try:
        result = Parallel(n_jobs=num_cores)(delayed(filter_pixel)(i, j) for i in range(N) for j in range(M))
    except:
        with parallel_backend('threading', n_jobs=num_cores):
            result = Parallel()(delayed(filter_pixel)(i, j) for i in range(N) for j in range(M))

    # Unpack the results 
    for (index, v), value in zip(np.ndenumerate(img_filtered), result):
        img_filtered[index[0], index[1]] = value

    if isinstance(final_img, xr.DataArray): # in case xarray dataarray was used
        final_img.values = img_filtered
        return final_img
    else:
        return img_filtered
