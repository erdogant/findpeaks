# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : MIT
# ----------------------------------------------------

import pandas as pd
import numpy as np
from peakdetect import peakdetect
# from findpeaks.utils.peakdetect import peakdetect
from findpeaks.utils.smoothline import smooth_line1d
import matplotlib.pyplot as plt

def fit(X, smooth=10, verbose=3):
    """Detection of peaks and valleys in a 1D vector.

    Parameters
    ----------
    X : array-like
        Input data.
    smooth : int, (default : 10)
        Smoothing factor. The higher the number, the more smoothing will occur.

    Returns
    -------
    dict.
    labx : array-like
        Labels of the detected distributions.
    max_peaks : list
        Detected peaks with maximum.
    min_peaks : list
        Detected peaks with minimum.

    Examples
    --------
    >>> import findpeaks
    >>> X = [10,11,9,23,21,11,45,20,11,12]
    >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
    >>> out = findpeaks.fit(X)
    >>> findpeaks.plot(out)

    """
    # Check datatype
    if isinstance(X, list):
        X=np.array(X)
    if isinstance(X, type(pd.DataFrame())):
        X=X.values

    # Here we extend the data by factor 3 interpolation and then we can nicely smoothen the data.
    Xo = X
    if smooth:
        X = smooth_line1d(X, nboost=len(X)*smooth, method=2, showfig=False)

    # Peak detect
    [max_peaks, min_peaks] = peakdetect(np.array(X), lookahead=1)

    # Check
    if min_peaks==[] or max_peaks==[]:
        if verbose>=3: print('[findpeaks] >No peaks detected.')
        return(None)

    [idx_peaks,_] = zip(*max_peaks)
    idx_peaks = np.array(list(idx_peaks))
    idx_valleys, _ = zip(*min_peaks)
    idx_valleys = np.append(np.array(list(idx_valleys)), len(X) - 1)
    idx_valleys = np.append(0, idx_valleys)

    # Group distribution
    labx_s = np.zeros((len(X))) * np.nan
    for i in range(0, len(idx_valleys)-1):
        labx_s[idx_valleys[i]:idx_valleys[i+1]+1] = i + 1

    # Scale back to original data
    min_peaks = np.minimum(np.ceil(((idx_valleys/len(X))*len(Xo))).astype(int), len(Xo) - 1)
    max_peaks =  np.minimum(np.ceil(((idx_peaks/len(X))*len(Xo))).astype(int), len(Xo) - 1)
    # Scaling is not accurate for indexing and therefore, a second wave of searching for peaks
    max_peaks_corr = []
    for max_peak in max_peaks:
        getrange=np.arange(np.maximum(max_peak-3,0),np.minimum(max_peak+3,len(Xo)))
        max_peaks_corr.append(getrange[np.argmax(Xo[getrange])])
    # Scaling is not accurate for indexing and therefore, a second wave of searching for peaks
    min_peaks_corr = []
    for min_peak in min_peaks:
        getrange=np.arange(np.maximum(min_peak-3,0),np.minimum(min_peak+3,len(Xo)))
        min_peaks_corr.append(getrange[np.argmin(Xo[getrange])])
    # Set the labels
    labx = np.zeros((len(Xo))) * np.nan
    for i in range(0, len(min_peaks)-1):
        labx[min_peaks[i]:min_peaks[i+1]+1] = i + 1

    # Store
    out = {}
    out['labx_s'] = labx_s
    out['min_peaks_s'] = idx_valleys
    out['max_peaks_s'] = idx_peaks
    out['X_s'] = X
    # Store based on original locations
    out['labx'] = labx
    out['min_peaks'] = min_peaks_corr
    out['max_peaks'] = max_peaks_corr
    out['X'] = Xo

    return(out)

# %%
def plot(out, figsize=(15,8)):
    # Make figure
    ax = _plot_original(out['X'], out['labx'], out['min_peaks'], out['max_peaks'], title='Original data', figsize=figsize)
    ax = _plot_original(out['X_s'], out['labx_s'], out['min_peaks_s'], out['max_peaks_s'], title='Interpolated data', figsize=figsize)

def _plot_original(X, labx, min_peaks, max_peaks, title=None, figsize=(15,8)):
    uilabx = np.unique(labx)
    uilabx = uilabx[~np.isnan(uilabx)]

    fig,ax = plt.subplots(figsize=figsize)
    plt.plot(X, 'k')
    plt.plot(max_peaks, X[max_peaks], "x", label='Top')
    plt.plot(min_peaks, X[min_peaks], "o", label='Bottom')

    # Color each detected label
    s=np.arange(0,len(X))
    for i in uilabx:
        idx=(labx==i)
        plt.plot(s[idx], X[idx], label='peak' + str(i))

    plt.legend(loc=0)
    plt.title(title)
    plt.grid(True)
