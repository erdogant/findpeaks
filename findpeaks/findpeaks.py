# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : See Licences
# ----------------------------------------------------
# from findpeaks.utils.utils import _compute_with_topology, _compute_with_mask
import findpeaks.utils.utils as utils
from findpeaks.utils.smoothline import smooth_line1d
from peakdetect import peakdetect
import cv2  # Only for 2D images required
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pandas as pd
import numpy as np
import wget
import os

class findpeaks():
    def __init__(self, lookahead=200, smooth=None, mask=0, resize=None, scale=True, togray=True, denoise=10, figsize=(15, 8), verbose=3):
        """Initialize findpeaks parameters.

        Parameters 1D
        -------------
        lookahead : int, (default : 200)
            Looking ahead for peaks. For very small 1d arrays, lets say up to 50 datapoints, set low numbers such as 1 or 2.
        smooth : int, (default : 10)
            Smoothing factor by interpolation. The higher the number, the more smoothing will occur.

        Parameters 2D-array
        -------------------
        X : array-like RGB or 2D-array
            Input image data.
        mask : float, (default : 0)
            Values <= mask are set as background.
        scale : bool, (default : False)
            Scaling in range [0-255] by img*(255/max(img))
        denoise : int, (default : 10 or None to disable)
            Denoising image, where the first value is the  filter strength. Higher value removes noise better, but removes details of image also.
        togray : bool, (default : False)
            Conversion to gray scale.
        resize : tuple, (default : None)
            Resize to desired (width,length).
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        """
        # Store in object
        self.lookahead = lookahead
        self.smooth = smooth
        self.mask = mask
        self.resize = resize
        self.scale = scale
        self.togray = togray
        self.denoise = denoise
        self.figsize = figsize
        self.verbose = verbose

    def fit(self, X, xs=None):
        """Detect peaks and valleys in a 1D vector or 2D-array (image).

        Parameters
        ----------
        X : array-like 1D vector
            Input data.
        xs : array-like 1D vector
            Coordinates of the x-axis

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
        >>> from findpeaks import findpeaks
        >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
        >>> fp = findpeaks(smooth=10, lookahead=1)
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # 2D array example
        >>> from findpeaks import findpeaks
        >>> X = fp.import_example('2dpeaks')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Image example
        >>> from findpeaks import findpeaks
        >>> fp = findpeaks(denoise=30, resize=(300,300))
        >>> X = fp.import_example('2dpeaks_image')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Plot each seperately
        >>> fp.plot_preprocessing()
        >>> fp.plot_mask()
        >>> fp.plot_peristence()
        >>> fp.plot_mesh()

        References
        ----------
        * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
        * https://www.sthu.org/code/codesnippets/imagepers.html

        """
        # Check datatype
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, type(pd.DataFrame())):
            X = X.values

        if len(X.shape)>1:
            if self.verbose>=3: print('[findpeaks] >2D array is detected, finding 2d peaks..')
            results = self.peaks2d(X)
        else:
            if self.verbose>=3: print('[findpeaks] >1D array is detected, finding 1d peaks..')
            results = self.peaks1d(X, xs=xs)

        return(results)

    # Find peaks in 1D vector
    def peaks1d(self, X, xs=None):
        # Here we extend the data by factor 3 interpolation and then we can nicely smoothen the data.
        Xo = X.copy()
        results = {}
        results['Xorig'] = Xo
        if self.smooth:
            X = smooth_line1d(X, nboost=len(X) * self.smooth, method=2, showfig=False)

        # Peak detect
        max_peaks, min_peaks = peakdetect(np.array(X), lookahead=self.lookahead)

        # Check
        if min_peaks==[] or max_peaks==[]:
            if self.verbose>=3: print('[findpeaks] >No peaks detected. Tip: try lowering lookahead value.')
            return(None)

        idx_peaks, _ = zip(*max_peaks)
        idx_peaks = np.array(list(idx_peaks))
        idx_valleys, _ = zip(*min_peaks)
        idx_valleys = np.append(np.array(list(idx_valleys)), len(X) - 1)
        idx_valleys = np.append(0, idx_valleys)

        # Group distribution
        labx_s = np.zeros((len(X))) * np.nan
        for i in range(0, len(idx_valleys) - 1):
            labx_s[idx_valleys[i]:idx_valleys[i + 1] + 1] = i + 1

        if self.smooth:
            # Scale back to original data
            min_peaks = np.minimum(np.ceil(((idx_valleys / len(X)) * len(Xo))).astype(int), len(Xo) - 1)
            max_peaks =  np.minimum(np.ceil(((idx_peaks / len(X)) * len(Xo))).astype(int), len(Xo) - 1)
            # Scaling is not accurate for indexing and therefore, a second wave of searching for peaks
            max_peaks_corr = []
            for max_peak in max_peaks:
                getrange = np.arange(np.maximum(max_peak - self.lookahead, 0), np.minimum(max_peak + self.lookahead, len(Xo)))
                max_peaks_corr.append(getrange[np.argmax(Xo[getrange])])
            # Scaling is not accurate for indexing and therefore, a second wave of searching for peaks
            min_peaks_corr = []
            for min_peak in min_peaks:
                getrange = np.arange(np.maximum(min_peak - self.lookahead, 0), np.minimum(min_peak + self.lookahead, len(Xo)))
                min_peaks_corr.append(getrange[np.argmin(Xo[getrange])])
            # Set the labels
            labx = np.zeros((len(Xo))) * np.nan
            for i in range(0, len(min_peaks) - 1):
                labx[min_peaks[i]:min_peaks[i + 1] + 1] = i + 1

            # Store based on original locations
            results['labx'] = labx
            results['xs'] = np.arange(0, len(Xo))
            results['min_peaks'] = np.c_[min_peaks_corr, Xo[min_peaks_corr]]
            results['max_peaks'] = np.c_[max_peaks_corr, Xo[max_peaks_corr]]

        # Store
        results_s, self.args = self._store1d(X, xs, results, idx_valleys, idx_peaks, labx_s)
        self.results = {**results, **results_s}

        return(self.results)

    # Find peaks in 2D-array
    def peaks2d(self, X):
        """Detect peaks and valleys in a 2D-array or image.

        Parameters
        ----------
        X : array-like 1D vector
            Input data.

        Returns
        -------
        results : dict
            Axis of figures are stored in the dictionary.

        Examples
        --------
        >>> # 2D array example
        >>> from findpeaks import findpeaks
        >>> X = fp.import_example('2dpeaks')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Image example
        >>> from findpeaks import findpeaks
        >>> fp = findpeaks(denoise=30, resize=(300,300))
        >>> X = fp.import_example('2dpeaks_image')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Plot each seperately
        >>> fp.plot_preprocessing()
        >>> fp.plot_mask()
        >>> fp.plot_peristence()
        >>> fp.plot_mesh()

        """
        if not self.togray and len(X.shape)==3: raise Exception('[findpeaks] >Error:  Topology method requires 2D array. Your input is 3D. Hint: set togray=True.')
        # Preprocessing the iamge
        Xproc = self.preprocessing(X, showfig=False)
        # Compute mesh-grid and persistance
        g0, xx, yy = utils._compute_with_topology(Xproc)
        # Compute peaks using local maximum filter.
        Xmask = utils._compute_with_mask(Xproc, mask=self.mask)
        # Store
        self.results, self.args = self._store2d(X, Xproc, Xmask, g0, xx, yy)
        # Return
        if self.verbose>=3: print('[findpeaks] >Fin.')
        return self.results

    # Store 1D vector
    def _store1d(self, X, xs, results, idx_valleys, idx_peaks, labx_s):
        results = {}
        if xs is None: xs = np.arange(0, len(X))
        results['labx_s'] = labx_s
        results['min_peaks_s'] = np.c_[idx_valleys, X[idx_valleys]]
        results['max_peaks_s'] = np.c_[idx_peaks, X[idx_peaks]]
        results['X_s'] = X
        results['xs_s'] = xs
        args = {}
        args['lookahead'] = self.lookahead
        args['smooth'] = self.smooth
        args['figsize'] = self.figsize
        args['method'] = 'peaks1d'

        return results, args

    # Store 2D-array
    def _store2d(self, X, Xproc, Xmask, g0, xx, yy):
        results = {}
        results['Xorig'] = X
        results['Xproc'] = Xproc
        results['Xmask'] = Xmask
        results['g0'] = g0
        results['xx'] = xx
        results['yy'] = yy
        args = {}
        args['mask'] = self.mask
        args['scale'] = self.scale
        args['denoise'] = self.denoise
        args['togray'] = self.togray
        args['resize'] = self.resize
        args['figsize'] = self.figsize
        args['method'] = 'peaks2d'
        return results, args

    # Pre-processing
    def preprocessing(self, X, showfig=None):
        """Preprocessing steps of the 2D array (image).

        Description
        -----------
        The pre-processing has 4 (optional) steps.
            1. Resizing (to reduce computation time).
            2. Scaling color pixels between [0-255]
            3. Conversion to gray-scale. This is required for some analysis.
            4. Denoising of the image.

        Parameters
        ----------
        X : numpy-array
            Input data or image.
        showfig : bool
            Show the preocessing steps in figures. The default is None.

        Returns
        -------
        X : numpy-array
            Processed image.

        """
        showfig = showfig if showfig is not None else self.showfig

        # Resize
        if self.resize:
            X = utils._resize(X, resize=self.resize)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
        # Scaling
        if self.scale:
            X = utils._scale(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
        # Convert to gray image
        if self.togray:
            X = utils._togray(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X, cmap=('gray' if self.togray else None))
        # Denoising
        if self.denoise is not None:
            X = utils._denoise(X, h=self.denoise, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X, cmap=('gray' if self.togray else None))
        # Return
        return X

    # %% Plotting
    def plot(self, figsize=None):
        """Plot results.

        Parameters
        ----------
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing (fig, ax)

        """
        figsize = figsize if figsize is not None else self.args['figsize']

        if not hasattr(self, 'results'):
            print('[findpeaks] Nothing to plot.')
            return
        elif self.args['method']=='peaks1d':
            fig_axis = self.plot1d(figsize=figsize)
        elif self.args['method']=='peaks2d':
            fig_axis = self.plot2d(figsize=figsize)
        # Return
        return fig_axis

    def plot1d(self, figsize=None):
        """Plot the 1D results.

        Parameters
        ----------
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        figsize = figsize if figsize is not None else self.args['figsize']
        ax1, ax2 = None, None
        # Make second plot
        if self.results.get('min_peaks', None) is not None:
            ax1 = _plot_original(self.results['Xorig'], self.results['xs'], self.results['labx'], self.results['min_peaks'][:, 0].astype(int), self.results['max_peaks'][:, 0].astype(int), title='Data', figsize=figsize)
        # Make smoothed plot
        ax2 = _plot_original(self.results['X_s'], self.results['xs_s'], self.results['labx_s'], self.results['min_peaks_s'][:, 0].astype(int), self.results['max_peaks_s'][:, 0].astype(int), title='Data', figsize=figsize)
        # Return axis
        return (ax2, ax1)

    def plot2d(self, figsize=None):
        """Plots the 2d results.

        Parameters
        ----------
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        figsize = figsize if figsize is not None else self.args['figsize']
        # Plot preprocessing steps
        self.plot_preprocessing()
        # Setup figure
        ax1, ax2, ax3 = self.plot_mask(figsize=figsize)
        # Plot persistence
        ax3, ax4 = self.plot_peristence(figsize=figsize)
        # Plot mesh
        ax5, ax6 = self.plot_mesh(figsize=figsize)
        # Return axis
        return (ax1, ax2, ax3, ax4, ax5, ax6)

    def plot_preprocessing(self):
        """Plot the pre-processing steps.

        Returns
        -------
        None.

        """
        # _ = self.preprocessing(self.results['Xorig'], mask=self.args['mask'], scale=self.args['scale'], denoise=self.args['denoise'], togray=self.args['togray'], resize=self.args['resize'], showfig=True, figsize=figsize, verbose=self.args['verbose'])
        _ = self.preprocessing(X=self.results['Xorig'], showfig=True)

    def plot_mask(self, figsize=None):
        """Plot the masking.

        Parameters
        ----------
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        figsize = figsize if figsize is not None else self.args['figsize']
        # Setup figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        # Original image
        cmap = 'gray' if self.args['togray'] else None

        # Plot input image
        ax1.imshow(self.results['Xorig'], cmap=cmap, interpolation="nearest")
        # ax1.invert_yaxis()
        ax1.set_title('Original')

        # Preprocessing
        ax2.imshow(self.results['Xproc'], cmap=cmap, interpolation="nearest")
        # ax2.invert_yaxis()
        ax2.set_title('Processed image')

        # Masking
        ax3.imshow(self.results['Xmask'], cmap=cmap, interpolation="nearest")
        # ax3.invert_yaxis()
        ax3.set_title('After Masking')

        # Return
        return (ax1, ax2, ax3)

    def plot_mesh(self, rstride=2, cstride=2, cmap=plt.cm.hot_r, figsize=None, view=None):
        """Plot the 3d-mesh.

        Parameters
        ----------
        rstride : int, (default is 2)
            Array row stride (step size).
        cstride : int, (default is 2)
            Array column stride (step size).
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.
        view : tuple, (default : None)
            Rotate the mesh plot.
            (0, 0) : y vs z
            (0, 90) : x vs z
            (90, 0) : y vs x
            (90, 90) : x vs y
        cmap : object
            Colormap. The default is plt.cm.hot_r.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not hasattr(self, 'results'):
            if self.verbose>=3: print('[findpeaks] >Nothing to plot. Hint: run the fit() function.')
        figsize = figsize if figsize is not None else self.args['figsize']
        if self.verbose>=3: print('[findpeaks] >Plotting 3d-mesh..')

        # Plot the figure
        fig = plt.figure(figsize=figsize)
        ax1 = fig.gca(projection='3d')
        ax1.plot_wireframe(self.results['xx'], self.results['yy'], self.results['Xproc'], rstride=rstride, cstride=cstride, linewidth=0.8)
        ax1.set_xlabel('x-axis')
        ax1.set_ylabel('y-axis')
        ax1.set_zlabel('z-axis')
        if view is not None:
            ax1.view_init(view[0], view[1])
            # ax1.view_init(50, -10) # x vs y
        plt.show()

        # Plot the figure
        fig = plt.figure(figsize=figsize)
        ax2 = fig.gca(projection='3d')
        ax2.plot_surface(self.results['xx'], self.results['yy'], self.results['Xproc'], rstride=rstride, cstride=cstride, cmap=cmap, linewidth=0, shade=True, antialiased=False)
        if view is not None:
            ax2.view_init(view[0], view[1])
        ax2.set_xlabel('x-axis')
        ax2.set_ylabel('y-axis')
        ax2.set_zlabel('z-axis')
        plt.show()

        # Plot with contours
        # fig = plt.figure(figsize=figsize)
        # ax3 = fig.gca(projection='3d')
        # X, Y, Z = results['xx'], results['yy'], results['Xproc']
        # ax3.plot_surface(results['xx'], results['yy'], results['Xproc'], rstride=rstride, cstride=cstride, cmap=plt.cm.coolwarm, linewidth=0, shade=True, alpha=0.3)
        # cset = ax3.contour(X, Y, Z, zdir='z', offset=-100, cmap=plt.cm.coolwarm)
        # cset = ax3.contour(X, Y, Z, zdir='x', offset=-40, cmap=plt.cm.coolwarm)
        # cset = ax3.contour(X, Y, Z, zdir='y', offset=40, cmap=plt.cm.coolwarm)
        # plt.show()
        return ax1, ax2

    def plot_peristence(self, figsize=None, verbose=3):
        figsize = figsize if figsize is not None else self.args['figsize']
        if not hasattr(self, 'results'):
            if verbose>=3: print('[findpeaks] >Nothing to plot. Hint: run the fit() function.')

        # Make the figure
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=figsize)
        # Plot the detected loci
        if verbose>=3: print('[findpeaks] >Plotting loci of birth..')
        ax1.set_title("Loci of births")
        for i, homclass in tqdm(enumerate(self.results['g0'])):
            p_birth, bl, pers, p_death = homclass
            if pers <= 20.0:
                continue
            y, x = p_birth
            ax1.plot([x], [y], '.', c='b')
            ax1.text(x, y + 0.25, str(i + 1), color='b')

        ax1.set_xlim((0, self.results['Xproc'].shape[1]))
        ax1.set_ylim((0, self.results['Xproc'].shape[0]))
        ax1.invert_yaxis()
        plt.gca().invert_yaxis()
        ax1.grid(True)

        # Plot the persistence
        if verbose>=3: print('[findpeaks] >Plotting Peristence..')
        ax2.set_title("Peristence diagram")
        ax2.plot([0, 255], [0, 255], '-', c='grey')
        for i, homclass in tqdm(enumerate(self.results['g0'])):
            p_birth, bl, pers, p_death = homclass
            if pers <= 1.0:
                continue

            x, y = bl, bl-pers
            ax2.plot([x], [y], '.', c='b')
            ax2.text(x, y + 2, str(i + 1), color='b')

        ax2.set_xlabel("Birth level")
        ax2.set_ylabel("Death level")
        ax2.set_xlim((-5, 260))
        ax2.set_ylim((-5, 260))
        ax2.grid(True)
        return ax1, ax2

    def import_example(self, data='2dpeaks', url=None, sep=';'):
        X = _import_example(data=data, url=url, sep=sep, verbose=self.verbose)
        return X

# %%
def _plot_original(X, xs, labx, min_peaks, max_peaks, title=None, figsize=(15, 8)):
    uilabx = np.unique(labx)
    uilabx = uilabx[~np.isnan(uilabx)]

    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(xs, X, 'k')
    plt.plot(max_peaks, X[max_peaks], "x", label='Top')
    plt.plot(min_peaks, X[min_peaks], "o", label='Bottom')

    # Color each detected label
    s=np.arange(0, len(X))
    for i in uilabx:
        idx=(labx==i)
        plt.plot(s[idx], X[idx], label='peak' + str(i))

    if len(uilabx) <= 10:
        plt.legend(loc=0)
    plt.title(title)
    plt.grid(True)
    return ax


# %% Import example dataset from github.
def _import_example(data='2dpeaks', url=None, sep=';', verbose=3):
    """Import example dataset from github source.

    Description
    -----------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: "2dpeaks" or "2dpeaks_image"
    url : str
        url link to to dataset.
    Verbose : int (default : 3)
        Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is not None:
        data = wget.filename_from_url(url)
    elif data=='2dpeaks_image':
        url='https://erdogant.github.io/datasets/' + data + '.png'
    elif data=='2dpeaks':
        url='https://erdogant.github.io/datasets/' + data + '.zip'
    else:
        if verbose>=3: print('[findpeaks] >Nothing to download <return>.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[findpeaks] >Downloading from github source: [%s]' %(url))
        wget.download(url, curpath)

    # Import local dataset
    if verbose>=3: print('[findpeaks] >Import [%s]' %(PATH_TO_DATA))
    if data=='2dpeaks_image':
        X = cv2.imread(PATH_TO_DATA)
    else:
        X = pd.read_csv(PATH_TO_DATA, sep=sep).values
    # Return
    return X
