# ----------------------------------------------------
# Name        : interpolate.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : MIT
# ----------------------------------------------------

import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# %%
def interpolate_line1d(X, n=3, method=2, showfig=False):
    """Interpolate 1d-vector.

    Parameters
    ----------
    X : array-like (1D-vector)
        Input image data.
    n : int, (default : 3)
        The interpolation factor. The data is interpolation by a factor n.
    method : str (default: 'linear')
         String or integer
         * 0 : order degree
         * 1 : order degree
         * 2 : order degree
         * 3 : order degree
         * 'linear' (default)
         * 'nearest'
         * 'zero'
         * 'slinear'
         * 'quadratic'
         * 'cubic'
         * 'previous'
         * 'next'
        showfig : bool, (default : False)
            Show the figure.

    Returns
    -------
    X : array-like (1D-vector)
        Interpolated data.

    """

    nboost = len(X) * n
    if len(X)>nboost: raise Exception('[findpeaks] >nboost (n=%.0f) must be larger then input data (n=%.0f)' %(nboost, len(X)))
    bootstdata = np.zeros(nboost) * np.nan
    idx = np.unique(np.floor(np.linspace(0, len(bootstdata) - 1, len(X))).astype(int))
    bootstdata[idx] = X

    X = interpolate_nans(bootstdata, method=method)
    logger.info('Interpolating 1d-vector by factor %d' %(n))

    if showfig:
        plot(X, bootstdata, method)

    return(X)

# %% Smooting of the line
def interpolate_line2d(xs, ys=None, interpol=3, window=3):
    """interpolate 2D vector.

    2D Interpolation Description
    ----------------------------
    Smoothing a 2d vector can be challanging if the data is low sampled.
    This function contains two steps. First interpolation of the input line followed by a convolution.

    Parameters
    ----------
    xs : array-like
        Data points for the x-axis.
    ys : array-like
        Data points for the y-axis.
    interpol : int, (default : 3)
        The interpolation factor. The data is interpolation by a factor n before the smoothing step.
    window : int, (default : 3)
        Smoothing window that is used to create the convolution and gradually smoothen the line.

    Returns
    -------
    xnew : array-like
        Data points for the x-axis.
    ynew : array-like
        Data points for the y-axis.

    """
    if window is not None:
        logger.info('Interpolating 2d-array (image) by factor %d' %(interpol))
        # Specify number of points to interpolate the data
        # Interpolate xs line
        extpoints = np.linspace(0, len(xs), len(xs) * interpol)
        spl = make_interp_spline(range(0, len(xs)), xs, k=3)
        xnew = spl(extpoints)
        xnew[window:-window]

        # First smoothing on the raw input data
        ynew=None
        if ys is not None:
            ys = _smooth(ys, window)
            # Interpolate ys line
            spl = make_interp_spline(range(0, len(ys)), ys, k=3)
            ynew = spl(extpoints)
            ynew[window:-window]
    else:
        xnew, ynew = xs, ys
    return xnew, ynew

def _smooth(X, window):
    box = np.ones(window) / window
    X_smooth = np.convolve(X, box, mode='same')
    return X_smooth

# %% Plot
def plot(X, bootstdata, method):
    plt.figure()
    # plt.plot(X, label='Boosted')
    plt.plot(interpolate_nans(bootstdata, method=0), label='0nd order')
    plt.plot(interpolate_nans(bootstdata, method=1), label='1nd order')
    plt.plot(interpolate_nans(bootstdata, method=2), label='2nd order')
    plt.plot(interpolate_nans(bootstdata, method=3), label='3nd order')
    plt.grid(True)
    plt.legend()


# %% interpolatie
def interpolate_nans(X, method='linear', replace_value_to_nan=None):
    """Interpolate the nan values in an 1D array.

    Parameters
    ----------
    X : array-like
        input data X.
    replace_value_to_nan : float (default: None)
        Replace value to np.nan.
        * None : take finite data: interplate np.nan.
        * 0 : take np.nan with the additional value (alternative)
    method : str (default: 'linear')
         String or integer
         * 0 : order degree
         * 1 : order degree
         * 2 : order degree
         * 3 : order degree
         * 'linear' (default)
         * 'nearest'
         * 'zero'
         * 'slinear'
         * 'quadratic'
         * 'cubic'
         * 'previous'
         * 'next'

    Returns
    -------
    None.

    Examples
    --------
    >>> X = np.array([1,2,3,np.nan,np.nan,6,7,np.nan,9,10])
    >>> Xint1 = interpolate_nans(X)
    >>> Xint2 = interpolate_nans(X, method=0)

    """
    yhat = []
    # Replace replace_value_to_nan by NaN value
    if replace_value_to_nan is not None: 
        X[X==replace_value_to_nan]=np.nan
    good = np.where(np.isfinite(X))[0]

    # interpolate to fill nan values
    inds = np.arange(X.shape[0])

    # Check for nan values
    if len(good) == 0:
        logger.warning('WARNING: Skipping because nothing to do: No nan values (?)')
        # yhat= np.nan_to_num(X)
        yhat = X
        # np.nan_to_num(X)
    else:
        f = interp1d(inds[good], X[good], bounds_error=False, kind=method)
        yhat = np.where(np.isfinite(X), X, f(inds))

    return(yhat)