# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : MIT
# ----------------------------------------------------
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
from scipy import misc
import findpeaks.utils.imagepers as imagepers
import numpy as np
import cv2 # Only for 2D images required

# %% Scaling
def _scale(X, verbose=3):
    if verbose>=3: print('[findpeaks] >Scaling image between [0-255] and to uint8')
    try:
        # Normalizing between 0-255
        X = X - X.min()
        X = X / X.max()
        X = X * 255
        # Downscale typing
        X = np.uint8(X)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Scaling not possible.')
    return X

# %%
def _togray(X, verbose=3):
    try:
        if verbose>=3: print('[findpeaks] >Conversion to gray image.')
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Conversion to gray not possible.')
    return X

# %%
def _denoise(X, h=10, verbose=3):
    try:
        if len(X.shape)==2:
            if verbose>=3: print('[findpeaks] >Denoising gray image.')
            X = cv2.fastNlMeansDenoising(X, h=h)
        if len(X.shape)==3:
            if verbose>=3: print('[findpeaks] >Denoising color image.')
            X = cv2.fastNlMeansDenoisingColored(X, h=h)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Denoising not possible.')
    return X

# %%
def _resize(X, resize=None, verbose=3):
    try:
        if resize is not None:
            if verbose>=3: print('[findpeaks] >Resizing image to %s.' %(str(resize)))
            X = cv2.resize(X, resize)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Resizing not possible.')
    return X

# %%
def _compute_with_topology(X, verbose=3):
    """Determine peaks in 2d-array using toplogy method.
    
    Description
    -----------
    A simple Python implementation of the 0-th dimensional persistent homology for 2D images.
    It is based on a two-dimensional persistent topology for peak detection.


    Parameters
    ----------
    X : numpy array
        2D array.

    Returns
    -------
    g0 : list
        Detected peaks.
    xx : numpy-array
        Meshgrid coordinates.
    yy : numpy-array
        Meshgrid coordinates.

    References
    ----------
    * https://www.sthu.org/code/codesnippets/imagepers.html
    * H. Edelsbrunner and J. Harer, Computational Topology. An Introduction, 2010, ISBN 0-8218-4925-5.

    """
    if verbose>=3: print('[findpeaks] >Detect peaks using topology method.')
    # Compute meshgrid
    xx, yy = np.mgrid[0:X.shape[0], 0:X.shape[1]]
    # Compute persistence
    g0 = imagepers.persistence(X)
    # Return
    return g0, xx, yy

# %%
def _compute_with_mask(X, mask=0, verbose=3):
    """Determine peaks in 2d-array using a mask.

    Description
    -----------
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Parameters
    ----------
    X : numpy array
        2D array.
    mask : float, (default : 0)
        Values <= mask are set as background.

    Returns
    -------
    detected_peaks : numpy array
        2D boolean array. True represents the detected peaks.

    References
    ----------
    * https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

    """
    if verbose>=3: print('[findpeaks] >Detect peaks using the masking (=%d) method.' %(mask))
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
    local_max = maximum_filter(X, footprint=neighborhood)==X
    # local_max is a mask that contains the peaks we are looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (X <= mask)

    # Erode the background in order to successfully subtract it form local_max,
    # otherwise a line will appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # We obtain the final mask, containing only peaks, by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    # Return
    return detected_peaks
