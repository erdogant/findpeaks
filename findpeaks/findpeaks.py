# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : See Licences
# ----------------------------------------------------

import findpeaks.utils.compute as compute
from findpeaks.utils.smoothline import interpolate_line1d
from peakdetect import peakdetect
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import wget
import os

class findpeaks():
    def __init__(self, lookahead=200, interpolate=None, mask=0, resize=None, scale=True, togray=True, denoise='fastnl', h=10, figsize=(15, 8), verbose=3):
        """Initialize findpeaks parameters.

        Parameters 1D
        -------------
        lookahead : int, (default : 200)
            Looking ahead for peaks. For very small 1d arrays, lets say up to 50 datapoints, set low numbers such as 1 or 2.
        interpolate : int, (default : 10)
            Interpoliation factor. The higher the number, the less sharp the edges will be.

        Parameters 2D-array
        -------------------
        X : array-like RGB or 2D-array
            Input image data.
        mask : float, (default : 0)
            Values <= mask are set as background.
        scale : bool, (default : False)
            Scaling in range [0-255] by img*(255/max(img))
        denoise : int, (default : 'bilateral', None to disable)
            'bilateral','fastnl' or None to denoise images.
        h : int, (default : 10)
            Denoising filter strength. Higher value removes noise better, but also removes details of image.
        togray : bool, (default : False)
            Conversion to gray scale.
        resize : tuple, (default : None)
            Resize to desired (width,length).
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        """
        # Store in object
        self.lookahead = lookahead
        self.interpolate = interpolate
        self.mask = mask
        self.resize = resize
        self.scale = scale
        self.togray = togray
        self.denoise = denoise
        self.h = h
        self.figsize = figsize
        self.verbose = verbose

    def fit(self, X, x=None):
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
        Xraw : array-like
            Input array
        X : array-like
            Processed array
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
        >>> fp = findpeaks(interpolate=10, lookahead=1)
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
            results = self.peaks1d(X, x=x)

        return(results)

    # Find peaks in 1D vector
    def peaks1d(self, X, x=None):
        Xraw = X.copy()
        # Interpolation
        if self.interpolate: X = interpolate_line1d(X, nboost=len(X) * self.interpolate, method=2, showfig=False)
        # Peak detect
        max_peaks, min_peaks = peakdetect(np.array(X), lookahead=self.lookahead)
        # Post processing
        results = compute._post_processing(X, Xraw, min_peaks, max_peaks, self.interpolate, self.lookahead)
        # Compute persistance using toplogy method
        persist_score = compute.persistence(np.c_[X, X])
        # Store
        self.results, self.args = self._store1d(X, Xraw, x, persist_score, results)
        # Return
        return(self.results)

    # Find peaks in 2D-array
    def peaks2d(self, X):
        """Detect peaks and valleys in a 2D-array or image.

        Description
        -----------
        Methodology. The idea behind the topology method: Consider the function graph of the function that assigns each pixel its level.
        Now consider a water level at height 255 that continuously descents to lower levels. At local maxima islands pop up (birth). At saddle points two islands merge; we consider the lower island to be merged to the higher island (death). The so-called persistence diagram (of the 0-th dimensional homology classes, our islands) depicts death- over birth-values of all islands.
        The persistence of an island is then the difference between the birth- and death-level; the vertical distance of a dot to the grey main diagonal. The figure labels the islands by decreasing persistence.
        The very first picture shows the locations of births of the islands. This method not only gives the local maxima but also quantifies their "significance" by the above mentioned persistence. One would then filter out all islands with a too low persistence. However, in your example every island (i.e., every local maximum) is a peak you look for.

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
        >>> X = fp.import_example('2dpeaks_image')
        >>> fp = findpeaks(denoise='fastnl', h=30, resize=(300,300))
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
        g0, xx, yy = compute._topology(Xproc)
        # Compute peaks using local maximum filter.
        Xmask = compute._mask(Xproc, mask=self.mask)
        # Store
        self.results, self.args = self._store2d(X, Xproc, Xmask, g0, xx, yy)
        # Return
        if self.verbose>=3: print('[findpeaks] >Fin.')
        return self.results

    # Store 1D vector
    def _store1d(self, X, Xraw, xs, persist_score, results):
        if xs is None: xs = np.arange(0, len(X))
        results['Xproc'] = X
        results['Xraw'] = Xraw
        results['x'] = xs
        results['persitance'] = persist_score
        # Arguments
        args = {}
        args['lookahead'] = self.lookahead
        args['interpolate'] = self.interpolate
        args['figsize'] = self.figsize
        args['method'] = 'peaks1d'
        # Return
        return results, args

    # Store 2D-array
    def _store2d(self, X, Xproc, Xmask, g0, xx, yy):
        results = {}
        results['Xraw'] = X
        results['Xproc'] = Xproc
        results['Xmask'] = Xmask
        results['persitance'] = g0
        results['xx'] = xx
        results['yy'] = yy
        # Store arguments
        args = {}
        args['mask'] = self.mask
        args['scale'] = self.scale
        args['denoise'] = self.denoise
        args['togray'] = self.togray
        args['resize'] = self.resize
        args['figsize'] = self.figsize
        args['method'] = 'peaks2d'
        # Return
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
            X = compute._resize(X, resize=self.resize)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
        # Scaling
        if self.scale:
            X = compute._scale(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
        # Convert to gray image
        if self.togray:
            X = compute._togray(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X, cmap=('gray' if self.togray else None))
        # Denoising
        if self.denoise is not None:
            X = compute.denoise(X, method=self.denoise, h=self.h, verbose=self.verbose)
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
        if not hasattr(self, 'results'):
            print('[findpeaks] Nothing to plot.')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']

        if self.args['method']=='peaks1d':
            fig_axis = self.plot1d(figsize=figsize)
        elif self.args['method']=='peaks2d':
            fig_axis = self.plot2d(figsize=figsize)
        else:
            print('[findpeaks] Nothing to plot for %s' %(self.args['method']))
            return None

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

        # Make plot
        if self.results.get('min_peaks', None) is not None:
            ax1 = _plot_original(self.results['Xraw'], self.results['xs'], self.results['labx'], self.results['min_peaks'][:, 0].astype(int), self.results['max_peaks'][:, 0].astype(int), title='Data', figsize=figsize)

        # Make interpolated plot
        if self.results.get('min_peaks_s', None) is not None:
            ax2 = _plot_original(self.results['Xproc'], self.results['x'], self.results['labx_s'], self.results['min_peaks_s'][:, 0].astype(int), self.results['max_peaks_s'][:, 0].astype(int), title='Data', figsize=figsize)
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
        # _ = self.preprocessing(self.results['Xraw'], mask=self.args['mask'], scale=self.args['scale'], denoise=self.args['denoise'], togray=self.args['togray'], resize=self.args['resize'], showfig=True, figsize=figsize, verbose=self.args['verbose'])
        _ = self.preprocessing(X=self.results['Xraw'], showfig=True)

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
        ax1.imshow(self.results['Xraw'], cmap=cmap, interpolation="nearest")
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
        for i, homclass in tqdm(enumerate(self.results['persitance'])):
            p_birth, bl, pers, p_death = homclass
            if pers <= 20.0:
                continue
            y, x = p_birth
            ax1.plot([x], [y], '.', c='b')
            ax1.text(x, y + 0.25, str(i + 1), color='b')
            
        if len(self.results['Xproc'].shape)>1:
            ax1.set_xlim((0, self.results['Xproc'].shape[1]))
            ax1.set_ylim((0, self.results['Xproc'].shape[0]))
            ax1.invert_yaxis()
            plt.gca().invert_yaxis()
            ax1.grid(True)

        # Plot the persistence
        if verbose>=3: print('[findpeaks] >Plotting Peristence..')
        ax2.set_title("Peristence diagram")
        ax2.plot([0, 255], [0, 255], '-', c='grey')
        for i, homclass in tqdm(enumerate(self.results['persitance'])):
            p_birth, bl, pers, p_death = homclass
            if pers <= 1.0:
                continue

            x, y = bl, bl-pers
            ax2.plot([x], [y], '.', c='b')
            ax2.text(x, y + 2, str(i + 1), color='b')

        ax2.set_xlabel("Birth level")
        ax2.set_ylabel("Death level")
        # ax2.set_xlim((-5, self.results['Xproc'].max().max()))
        # ax2.set_ylim((-5, self.results['Xproc'].max().max()))
        ax2.set_xlim((self.results['Xproc'].min().min(), self.results['Xproc'].max().max()))
        ax2.set_ylim((self.results['Xproc'].min().min(), self.results['Xproc'].max().max()))



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
    elif data=='1dpeaks':
        x=[0,   13,  22,  30,  35,  38,   42,   51,   57,   67,  73,   75,  89,   126,  141,  150,  200 ]
        y=[1.5, 0.8, 1.2, 0.2, 0.4, 0.39, 0.42, 0.22, 0.23, 0.1, 0.11, 0.1, 0.14, 0.09, 0.04,  0.02, 0.01]
        X = np.c_[x,y]
        return X
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
        cv2 = compute._import_cv2()
        X = cv2.imread(PATH_TO_DATA)
    else:
        X = pd.read_csv(PATH_TO_DATA, sep=sep).values
    # Return
    return X

