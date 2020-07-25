# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : MIT
# ----------------------------------------------------
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter, uniform_filter
from scipy import misc
import findpeaks.utils.imagepers as imagepers
import numpy as np

# Import cv2
def _import_cv2():
    # Only for 2D images required
    try:
        import cv2
        return cv2
    except:
        raise ImportError('cv2 must be installed manually. Try to: <pip install opencv-python>')

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
    # Import cv2
    cv2 = _import_cv2()
    try:
        if verbose>=3: print('[findpeaks] >Conversion to gray image.')
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Conversion to gray not possible.')
    return X

# %%
def lee_filter(X, window, var_noise=0.25):
    """Lee filter for speckle noise removal

    Description
    -----------
    The Additive Noise Lee Despeckling Filter
    Let's assume that the despeckling noise is additive with a constant mean of zero, a constant variance, and drawn from a Gaussian distribution. Use a window (I x J pixels) to scan the image with a stride of 1 pixels (and I will use reflective boundary conditions). The despeckled value of the pixel in the center of the window located in the ith row and jth column is, zhat_ij = mu_k + W*(z_ij = mu_z), where mu_k is the mean value of all pixels in the window centered on pixel i,j, z_ij is the unfiltered value of the pixel, and W is a weight calculated as, W = var_k / (var_k + var_noise), where var_k is the variance of all pixels in the window and var_noise is the variance of the speckle noise. A possible alternative to using the actual value of the center pixel for z_ij is to use the median pixel value in the window.
    The parameters of the filter are the window/kernel size and the variance of the noise (which is unknown but can perhaps be estimated from the image as the variance over a uniform feature smooth like the surface of still water). Using a larger window size and noise variance will increase radiometric resolution at the expense of spatial resolution.
    For more info on the Lee Filter and other despeckling filters see http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/speckle-function.htm
    Assumes noise mean = 0

    If you don't want the window to be a square of size x size, just replace uniform_filter with something else (convolution with a disk, gaussian filter, etc). Any type of (weighted) averaging filter will do, as long as it is the same for calculating both img_mean and img_square_mean.
    The Lee filter seems rather old-fashioned as a filter. It won't behave well at edges because for any window that has an edge in it, the variance is going to be much higher than the overall image variance, and therefore the weights (of the unfiltered image relative to the filtered image) are going to be close to 1.

    Parameters
    ----------
    X : TYPE
        SAR data to be despeckled (already reshaped into image dimensions).
    window : (tuple)
        descpeckling filter window.

    Returns
    -------
    band_filtered : TYPE
        DESCRIPTION.

    """
    from scipy.ndimage.measurements import variance

    img_mean = uniform_filter(X, (window, window))
    img_sqr_mean = uniform_filter(X**2, (window, window))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(X)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (X - img_mean)
    return img_output

    # mean_window = uniform_filter(X, window)
    # mean_sqr_window = uniform_filter(X**2, window)
    # var_window = mean_sqr_window - mean_window**2

    # weights = var_window / (var_window + var_noise)
    # X_filtered = mean_window + weights*(X - mean_window)
    # return X_filtered

# %%
def denoise(X, h=9, method='bilateral', verbose=3):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    # The bilateral filter uses a Gaussian filter in the space domain, 
    # but it also uses one more (multiplicative) Gaussian filter component which is a function of pixel intensity differences.
    # The Gaussian function of space makes sure that only pixels are ‘spatial neighbors’ are considered for filtering,
    # while the Gaussian component applied in the intensity domain (a Gaussian function of intensity differences)
    # ensures that only those pixels with intensities similar to that of the central pixel (‘intensity neighbors’)
    # are included to compute the blurred intensity value. As a result, this method preserves edges, since for pixels lying near edges,
    # neighboring pixels placed on the other side of the edge, and therefore exhibiting large intensity variations when
    # compared to the central pixel, will not be included for blurring.

    # Import library
    cv2 = _import_cv2()

    # Peform the denoising
    try:
        if verbose>=3: print('[findpeaks] >Denoising with [%s] and filter strength: %d.' %(method, h))
        if method=='fastnl':
            if len(X.shape)==2:
                X = cv2.fastNlMeansDenoising(X, h=h)
            if len(X.shape)==3:
                if verbose>=3: print('[findpeaks] >Denoising color image.')
                X = cv2.fastNlMeansDenoisingColored(X, h=h)
        elif method=='bilateral':
            X = cv2.bilateralFilter(X, h, 75, 75)
        elif method=='lee':
            X = lee_filter(X, h, var_noise=0.25)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Denoising not possible.')
    return X

# %%
def _resize(X, resize=None, verbose=3):
    # Import cv2
    cv2 = _import_cv2()
    try:
        if resize is not None:
            if verbose>=3: print('[findpeaks] >Resizing image to %s.' %(str(resize)))
            X = cv2.resize(X, resize)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Resizing not possible.')
    return X

# %%
def _topology(X, verbose=3):
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
def _mask(X, mask=0, verbose=3):
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
