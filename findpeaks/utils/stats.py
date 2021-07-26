# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : MIT
# ----------------------------------------------------

import findpeaks.utils.union_find as union_find
from findpeaks.filters.lee import lee_filter
from findpeaks.filters.lee_enhanced import lee_enhanced_filter
from findpeaks.filters.kuan import kuan_filter
from findpeaks.filters.frost import frost_filter
from findpeaks.filters.median import median_filter
from findpeaks.filters.mean import mean_filter

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter, uniform_filter
from scipy import misc

from tqdm import tqdm
import numpy as np
import pandas as pd

# %% Import cv2
def _import_cv2():
    # Only for 2D images required
    try:
        import cv2
        return cv2
    except:
        raise ImportError('cv2 must be installed manually. Try to: <pip install opencv-python>')


# %% Scaling
def scale(X, verbose=3):
    """Normalize data (image) by scaling.

    Description
    -----------
    Scaling in range [0-255] by img*(255/max(img))

    Parameters
    ----------
    X : array-like
        Input image data.
    verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    X : array-like
        Scaled image.

    """
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
def togray(X, verbose=3):
    """Convert color to grey-image.

    Description
    -----------
    Convert 3d-RGB colors to 2d-grey image.

    Parameters
    ----------
    X : array-like
        Input image data.
    verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    X : array-like
        2d-image.

    """
    # Import cv2
    cv2 = _import_cv2()
    try:
        if verbose>=3: print('[findpeaks] >Conversion to gray image.')
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Conversion to gray not possible.')
    return X


# %%
def resize(X, size=None, verbose=3):
    """Resize image.

    Parameters
    ----------
    X : array-like
        Input image data.
    size : tuple, (default : None)
        size to desired (width,length).
    verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    X : array-like

    """
    # Import cv2
    cv2 = _import_cv2()
    try:
        if size is not None:
            if verbose>=3: print('[findpeaks] >Resizing image to %s.' %(str(size)))
            X = cv2.resize(X, size)
    except:
        if verbose>=2: print('[findpeaks] >Warning: Resizing not possible.')
    return X


# %%
def denoise(X, method='fastnl', window=9, cu=0.25, verbose=3):
    """Denoise input data.

    Description
    -----------
    Denoising the data is very usefull before detection of peaks. Multiple methods are implemented to denoise the data.
    The bilateral filter uses a Gaussian filter in the space domain,
    but it also uses one more (multiplicative) Gaussian filter component which is a function of pixel intensity differences.
    The Gaussian function of space makes sure that only pixels are ‘spatial neighbors’ are considered for filtering,
    while the Gaussian component applied in the intensity domain (a Gaussian function of intensity differences)
    ensures that only those pixels with intensities similar to that of the central pixel (‘intensity neighbors’)
    are included to compute the blurred intensity value. As a result, this method preserves edges, since for pixels lying near edges,
    neighboring pixels placed on the other side of the edge, and therefore exhibiting large intensity variations when
    compared to the central pixel, will not be included for blurring.

    Parameters
    ----------
    X : array-like
        Input image data.
    method : string, (default : 'fastnl', None to disable)
        Filtering method to remove noise
            * None
            * 'fastnl'
            * 'bilateral'
            * 'lee'
            * 'lee_enhanced'
            * 'kuan'
            * 'frost'
            * 'median'
            * 'mean'
    window : int, (default : 3)
        Denoising window. Increasing the window size may removes noise better but may also removes details of image in certain denoising methods.
    cu : float, (default: 0.25)
        The noise variation coefficient, applies for methods: ['kuan','lee','lee_enhanced']
    verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    X : array-like
        Denoised data.

    References
    ----------
    * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    """
    if window is None: window=9
    if cu is None: cu=0.25
    # Import library
    cv2 = _import_cv2()

    # Peform the denoising
    # try:
    if verbose>=3: print('[findpeaks] >Denoising with [%s], window: [%d].' %(method, window))
    if method=='fastnl':
        if len(X.shape)==2:
            X = cv2.fastNlMeansDenoising(X, h=window)
        if len(X.shape)==3:
            if verbose>=3: print('[findpeaks] >Denoising color image.')
            X = cv2.fastNlMeansDenoisingColored(X, h=window)
    elif method=='bilateral':
        X = cv2.bilateralFilter(X, window, 75, 75)
    elif method=='lee':
        X = lee_filter(X, win_size=window, cu=cu)
    elif method=='lee_enhanced':
        X = lee_enhanced_filter(X, win_size=window, cu=cu, k=1, cmax=1.73)
    elif method=='kuan':
        X = kuan_filter(X, win_size=window, cu=cu)
    elif method=='frost':
        X = frost_filter(X, win_size=window, damping_factor=2)
    elif method=='median':
        X = median_filter(X, win_size=window)
    elif method=='mean':
        X = mean_filter(X, win_size=window)
    # except:
    #     if verbose>=2: print('[findpeaks] >Warning: Denoising failt!')
    return X


# %%
def mask(X, limit=0, verbose=3):
    """Determine peaks in 2d-array using a mask.

    Description
    -----------
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)

    Parameters
    ----------
    X : array-like
        Input image data.
    limit : float, (default : None)
        Values > limit are set as regions of interest (ROI).

    Returns
    -------
    dict()
        Xraw : array-like.
            Input image.
        Xdetect : array-like (same shape as input data)
            detected peaks with respect the input image. Elements are the scores.
        Xranked : array-like (same shape as input data)
            detected peaks with respect the input image. Elements are the ranked peaks (1=best).

    References
    ----------
    * https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

    """
    if limit is None: limit=0

    if verbose>=3: print('[findpeaks] >Detect peaks using the mask method with limit=%s.' %(limit))
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
    local_max = maximum_filter(X, footprint=neighborhood)==X
    # local_max is a mask that contains the peaks we are looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (X <= limit)

    # Erode the background in order to successfully subtract it form local_max,
    # otherwise a line will appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # We obtain the final mask, containing only peaks, by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    # Return
    return detected_peaks


def topology(X, limit=None, verbose=3):
    """Determine peaks in 2d-array using toplogy method.

    Description
    -----------
    The idea behind the topology method: Consider the function graph of the function that assigns each pixel its level.
    Now consider a water level that continuously descents to lower levels. At local maxima islands pop up (birth). At saddle points two islands merge; we consider the lower island to be merged to the higher island (death). The so-called persistence diagram (of the 0-th dimensional homology classes, our islands) depicts death- over birth-values of all islands.
    The persistence of an island is then the difference between the birth- and death-level; the vertical distance of a dot to the grey main diagonal. The figure labels the islands by decreasing persistence.
    This method not only gives the local maxima but also quantifies their "significance" by the above mentioned persistence. One would then filter out all islands with a too low persistence. However, in your example every island (i.e., every local maximum) is a peak you look for.

    Parameters
    ----------
    X : array-like data
        Input data.
    limit : float, (default : None)
        score > limit are set as regions of interest (ROI).
    verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    dict()
        Xdetect : array-like (same shape as input data)
            detected peaks with respect the input image. Elements are the scores.
        Xranked : array-like (same shape as input data)
            detected peaks with respect the input image. Elements are the ranked peaks (1=best).
        max_peaks : array-like
            Detected peaks
        min_peaks : array-like
            Detected vallyes
        persistence : DataFrame()
            * x, y    : coordinates
            * birth   : Birth level
            * death   : Death level
            * score   : persistence scores

    References
    ----------
    * https://www.sthu.org/code/codesnippets/imagepers.html
    * H. Edelsbrunner and J. Harer, Computational Topology. An Introduction, 2010, ISBN 0-8218-4925-5.
    * Initial implementation: Stefan Huber <shuber@sthu.org>
    * Editted by: Erdogan Taskesen <erdogant@gmail.com>

    """
    if verbose>=3: print('[findpeaks] >Detect peaks using topology method with limit at %s.' %(limit))

    h, w = X.shape
    max_peaks, min_peaks = None, None
    groups0 = {}

    # Get indices orderd by value from high to low
    indices = [(i, j) for i in range(h) for j in range(w)]
    indices.sort(key=lambda p: _get_indices(X, p), reverse=True)

    # Maintains the growing sets
    uf = union_find.UnionFind()

    def _get_comp_birth(p):
        return _get_indices(X, uf[p])

    # Process pixels from high to low
    disable = (True if ((verbose==0 or verbose is None) or verbose>3) else False)
    for i, p in tqdm(enumerate(indices), disable=disable):
        v = _get_indices(X, p)
        ni = [uf[q] for q in _iter_neighbors(p, w, h) if q in uf]
        nc = sorted([(_get_comp_birth(q), q) for q in set(ni)], reverse=True)

        if i == 0: groups0[p] = (v, v, None)
        uf.add(p, -i)

        if len(nc) > 0:
            oldp = nc[0][1]
            uf.union(oldp, p)
            # Merge all others with oldp
            for bl, q in nc[1:]:
                if uf[q] not in groups0:
                    groups0[uf[q]] = (bl, bl - v, p)
                uf.union(oldp, q)

    groups0 = [(k, groups0[k][0], groups0[k][1], groups0[k][2]) for k in groups0]
    groups0.sort(key=lambda g: g[2], reverse=True)

    # Filter on limit
    if (limit is not None):
        Ikeep = np.array(list(map(lambda x: x[2], groups0))) > limit
        groups0 = np.array(groups0, dtype='object')
        groups0 = groups0[Ikeep].tolist()

    # Extract the max peaks and sort
    max_peaks = np.array(list(map(lambda x: [x[0][0], x[1]], groups0)))
    idxsort = np.argsort(max_peaks[:, 0])
    max_peaks = max_peaks[idxsort, :]
    # Extract the min peaks and sort
    min_peaks = np.array(list(map(lambda x: [(x[3][0] if x[3] is not None else 0), x[2]], groups0)))
    idxsort = np.argsort(min_peaks[:, 0])
    min_peaks = min_peaks[idxsort, :].tolist()

    # Build the output results in the same manner as the input image
    Xdetect = np.zeros_like(X).astype(float)
    Xranked = np.zeros_like(X).astype(int)
    for i, homclass in enumerate(groups0):
        p_birth, bl, pers, p_death = homclass
        y, x = p_birth
        Xdetect[y, x] = pers
        Xranked[y, x] = i + 1

    # If data is 1d-vector, make single vector
    if (X.shape[1]==2) and (np.all(Xdetect[:, 1]==0)):
        Xdetect = Xdetect[:, 0]
        Xranked = Xranked[:, 0]

    # Store in dataframe
    df_persistence = pd.DataFrame()
    df_persistence['x'] = np.array(list(map(lambda x: x[0][0], groups0)))
    df_persistence['y'] = np.array(list(map(lambda x: x[1], groups0)))
    df_persistence['birth_level'] = np.array(list(map(lambda x: x[1], groups0)))
    df_persistence['death_level'] = np.array(list(map(lambda x: x[1] - x[2], groups0)))
    df_persistence['score'] = np.array(list(map(lambda x: x[2], groups0)))
    # Results
    results = {}
    results['groups0'] = groups0
    results['Xdetect'] = Xdetect
    results['Xranked'] = Xranked
    results['peak'] = max_peaks
    results['valley'] = min_peaks
    results['persistence'] = df_persistence

    # return
    return results


def _get_indices(im, p):
    return im[p[0]][p[1]]


def _iter_neighbors(p, w, h):
    y, x = p

    # 8-neighborship
    neigh = [(y + j, x + i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    # 4-neighborship
    # neigh = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

    for j, i in neigh:
        if j < 0 or j >= h:
            continue
        if i < 0 or i >= w:
            continue
        if j == y and i == x:
            continue
        yield j, i


def _post_processing(X, Xraw, min_peaks, max_peaks, interpolate, lookahead, labxRaw=None, verbose=3):
    if lookahead<1: raise Exception('[findpeaks] >lookhead parameter should be at least 1.')
    labx_s = np.zeros((len(X))) * np.nan
    results = {}
    results['min_peaks_s'] = None
    results['max_peaks_s'] = None
    results['xs'] = np.arange(0, len(Xraw))
    results['labx_s'] = labx_s
    results['labx'] = np.zeros((len(Xraw))) * np.nan
    results['min_peaks'] = None
    results['max_peaks'] = None
    
    if len(min_peaks)>0 and len(max_peaks)>0 and (max_peaks[0][0] is not None):

        idx_peaks, _ = zip(*max_peaks)
        idx_peaks = np.array(list(idx_peaks)).astype(int)
        idx_valleys, _ = zip(*min_peaks)
        idx_valleys = np.append(np.array(list(idx_valleys)), len(X) - 1).astype(int)
        idx_valleys = np.append(0, idx_valleys)

        # Group distribution
        count=1
        for i in range(0, len(idx_valleys) - 1):
            if idx_valleys[i]!=idx_valleys[i + 1]:
                labx_s[idx_valleys[i]:idx_valleys[i + 1] + 1] = count
                count=count + 1

        # Scale back to original data
        if interpolate is not None:
            min_peaks = np.minimum(np.ceil(((idx_valleys / len(X)) * len(Xraw))).astype(int), len(Xraw) - 1)
            max_peaks = np.minimum(np.ceil(((idx_peaks / len(X)) * len(Xraw))).astype(int), len(Xraw) - 1)
            # Scaling is not accurate for indexing and therefore, a second wave of searching for max_peaks
            max_peaks_corr = []
            for max_peak in max_peaks:
                getrange = np.arange(np.maximum(max_peak - lookahead, 0), np.minimum(max_peak + lookahead, len(Xraw)))
                max_peaks_corr.append(getrange[np.argmax(Xraw[getrange])])
            # Scaling is not accurate for indexing and therefore, a second wave of searching for min_peaks
            min_peaks_corr = []
            for min_peak in min_peaks:
                getrange = np.arange(np.maximum(min_peak - lookahead, 0), np.minimum(min_peak + lookahead, len(Xraw)))
                min_peaks_corr.append(getrange[np.argmin(Xraw[getrange])])
            # Set the labels
            count = 1
            labx = np.zeros((len(Xraw))) * np.nan
            for i in range(0, len(min_peaks) - 1):
                if min_peaks[i]!=min_peaks[i + 1]:
                    labx[min_peaks[i]:min_peaks[i + 1] + 1] = count
                    count=count + 1

            # Store based on original
            results['labx'] = labx
            results['min_peaks'] = np.c_[min_peaks_corr, Xraw[min_peaks_corr]]
            results['max_peaks'] = np.c_[max_peaks_corr, Xraw[max_peaks_corr]]

        results['min_peaks_s'] = np.c_[idx_valleys, X[idx_valleys]]
        results['max_peaks_s'] = np.c_[idx_peaks, X[idx_peaks]]
        if labxRaw is None:
            results['labx_s'] = labx_s
        else:
            results['labx_s'] = labxRaw

    # Return
    return results
