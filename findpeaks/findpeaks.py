"""Python library for the detection of peaks and valleys."""
# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : See LICENSE
# ----------------------------------------------------

# import findpeaks
from peakdetect import peakdetect
from caerus import caerus
import caerus.utils.csplots as csplots
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import requests
from urllib.parse import urlparse

# #### DEBUG ONLY ####
# import stats as stats
# from stats import disable_tqdm
# import interpolate as interpolate
# #####################
import findpeaks.stats as stats
from findpeaks.stats import disable_tqdm
import findpeaks.interpolate as interpolate
# #####################


# %%
class findpeaks():
    """Python library for the detection of peaks and valleys.

    findpeaks is for the detection and vizualization of peaks and valleys in a 1D-vector and 2D-array.
    In case of 2D-array, the image can be pre-processed by resizing, scaling, and denoising. For a 1D-vector,
    pre-processing by interpolation is possible. Peaks can be detected using various methods, and the results can be
    vizualized, such as the preprocessing steps, the persistence of peaks, the masking plot and a 3d-mesh plot.

    Examples
    --------
    >>> from findpeaks import findpeaks
    >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
    >>> fp = findpeaks(method='peakdetect', interpolate=10, lookahead=1)
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
    >>> fp = findpeaks(method='topology', denoise='fastnl', params={'window': 30}, imsize=(300,300))
    >>> X = fp.import_example('2dpeaks_image')
    >>> results = fp.fit(X)
    >>> fp.plot()
    >>>
    >>> # Plot each seperately
    >>> fp.plot_preprocessing()
    >>> fp.plot_persistence()
    >>> fp.plot_mesh()

    References
    ----------
        * https://erdogant.github.io/findpeaks/

    """

    def __init__(self,
                 method=None,
                 whitelist=['peak', 'valley'],
                 lookahead=200,
                 interpolate=None,
                 limit=0,
                 imsize=None,
                 scale=True,
                 togray=True,
                 denoise='fastnl',
                 window=None,  # DEPRECATED IN LATER VERSIONS: specify in params
                 cu=None,  # DEPRECATED IN LATER VERSIONS: specify in params
                 params_caerus={},  # DEPRECATED IN LATER VERSIONS: use params instead
                 params={'window': 3, 'delta': 0},
                 figsize=(15, 8),
                 verbose=3):
        """Initialize findpeaks parameters.

        Parameters
        ----------
        X : array-like (1d-vector or 2d-image)
            Input image data.
        method : String, (default : None).
            Available methods for peak detection. In case method=None, the default is choosen.
            1d-vector approaches:
                * 'topology'
                * 'peakdetect' (default)
                * 'caerus'
            2d-array approaches:
                * 'topology' (default)
                * 'mask'
        whitelist : str or list ['peak','valley']
            Choose what to detect:
                * 'peak'
                * 'valley'
                * ['peak','valley']
        lookahead : int, (default : 200)
            Looking ahead for peaks. For very small 1d arrays (such as up to 50 datapoints), use low numbers such as 1 or 2.
        interpolate : int, (default : None)
            Interpolation factor. The higher the number, the less sharp the edges will be.
        limit : float, (default : None)
            In case method='topology'
            Values > limit are active search areas to detect regions of interest (ROI).
        imsize : tuple, (default : None)
            resize to (width,length).
        scale : bool, (default : False)
            Scaling in range [0-255] by img*(255/max(img))
        denoise : string, (default : 'fastnl', None to disable)
            Filtering method to remove noise:
                * None
                * 'fastnl'
                * 'bilateral'
                * 'lee'
                * 'lee_enhanced'
                * 'lee_sigma'
                * 'kuan'
                * 'frost'
                * 'median'
                * 'mean'
        params : dict():
            Denoising parameters for the methods. If None are defined, the default will be used:
            * caerus (default): {'window': 50, 'minperc': 3, 'nlargest': 10, 'threshold': 0.25}
            * lee_sigma (default): {'window': 7, 'sigma': 0.9, 'num_looks': 1, 'tk': 5}
                * 'sigma': float, (default: 0.9): Speckle noise standard deviation, applies for methods: ['lee_sigma']
                * 'num_looks': int, (default: 1): Number of looks of the SAR img, applies for methods: ['lee_sigma']
                * 'tk': int, (default: 5): Threshold of neighbouring pixels outside of the 98th percentile, applies for methods: ['lee_sigma']
                * cu : float, (default: 0.25): The noise variation coefficient, applies for methods: ['kuan','lee','lee_enhanced']
                * window : int, (default : 3): Denoising window. Increasing the window size may removes noise better but may also removes details of image in certain denoising methods.
            * peakdetect
                'delta' : int (default: 0): this specifies a minimum difference between a peak and the following points, before a peak may be considered a peak. Useful to hinder the function
                from picking up false peaks towards to end of the signal. To work well delta should be set to delta >= RMSnoise * 5.
                When omitted delta function causes a 20% decrease in speed. When used Correctly it can double the speed of the function
        togray : bool, (default : False)
            Conversion to gray scale.
        verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Returns
        -------
        dict()
            See 1dpeaks and 2dpeaks for more details.

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
        >>> fp = findpeaks(method='peakdetect', interpolate=10, lookahead=1)
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
        >>> fp = findpeaks(method='topology', denoise='fastnl', params={'window': 30}, imsize=(300,300))
        >>> X = fp.import_example('2dpeaks_image')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Plot each seperately
        >>> fp.plot_preprocessing()
        >>> fp.plot_persistence()
        >>> fp.plot_mesh()

        References
        ----------
            * https://erdogant.github.io/findpeaks/
        """
        if window is not None: print('The input parameter "window" will be deprecated in future releases. Please use "params={"window": 5}" instead.')
        if cu is not None: print('The input parameter "cu" will be deprecated in future releases. Please use "params={"cu": 3}" instead.')

        # Store in object
        if isinstance(whitelist, str): whitelist=[whitelist]
        if lookahead is None: lookahead=1
        lookahead = np.maximum(1, lookahead)
        # if method is None: raise Exception('[findpeaks] >Specify the desired method="topology", "peakdetect", or "mask".')
        self.method = method
        self.whitelist = whitelist
        self.lookahead = lookahead
        self.interpolate = interpolate
        self.limit = limit
        self.imsize = imsize
        self.scale = scale
        self.togray = togray
        self.denoise = denoise
        self.figsize = figsize
        self.verbose = verbose

        # Store parameters for caerus
        defaults={}
        if method=='caerus':
            if len(params_caerus)>0:
                print('The input parameter "params_caerus" will be deprecated in future releases. Please use "params" instead.')
                params = params_caerus
            defaults = {'window': 50, 'minperc': 3, 'nlargest': 10, 'threshold': 0.25}
        elif method=='lee_sigma':
            defaults = {'window': 7, 'sigma': 0.9, 'num_looks': 1, 'tk': 5}
        elif method=='peakdetect':
            defaults = {'delta': 0}
        defaults = {**{'window': 3}, **defaults}

        params = {**defaults, **params}
        self.window = params['window']
        self.cu = params.get('cu', 0.25)
        self.params = params

    def fit(self, X, x=None):
        """Detect peaks and valleys in a 1D vector or 2D-array (image).

        Description
        -----------
        * Fit the method on your data for the detection of peaks.
        * See 1dpeaks and 2dpeaks for more details about the input/output parameters.

        Parameters
        ----------
        X : array-like data.
            Input data.
        x : array-like data.
            Coordinates of the x-axis.

        Returns
        -------
        dict()
            * See 1dpeaks and 2dpeaks for more details.

        """
        # Check datatype
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, type(pd.DataFrame())):
            X = X.values

        if len(X.shape)>1:
            # 2d-array (image)
            results = self.peaks2d(X, method=self.method)
        else:
            # 1d-array (vector)
            results = self.peaks1d(X, x=x, method=self.method)

        return results

    # Find peaks in 1D vector
    def peaks1d(self, X, x=None, method='peakdetect'):
        """Detect of peaks in 1D array.

        Description
        -----------
        This function only eats the input data. Use the .fit() function for more information regarding the input parameters:
            * method : method to be used for peak detection: 'topology' or 'peakdetect'.
            * lookahead : Looking ahead for peaks. For very small 1d arrays (such as up to 50 datapoints), use low numbers: 1 or 2.
            * interpolate : Interpolation factor. The higher the number, the less sharp the edges will be.
            * limit : Values > limit are set as regions of interest (ROI).
            * verbose : Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Parameters
        ----------
        X : array-like 1D vector.
            Input data.
        x : array-like 1D vector.
            Coordinates of the x-axis.

        Returns
        -------
        dict() : Results in "df" are based on the input-data, whereas "df_interp" are the interpolated results.
            * persistence : Scores when using topology method.
            * Xranked     : Similar to column "rank".
            * Xdetect     : Similar to the column "score".
            * df          : Is ranked in the same manner as the input data and provides information about the detected peaks and valleys.
        persistence : pd.DataFrame()
            * x, y    : coordinates
            * birth   : Birth level
            * death   : Death level
            * score   : persistence scores
        df : pd.DataFrame()
            * x, y    : Coordinates
            * labx    : The label of the peak area
            * rank    : The ranking number of the best performing peaks (1 is best)
            * score   : persistence score
            * valley  : Whether the point is marked as valley
            * peak    : Whether the point is marked as peak

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
        >>> fp = findpeaks(method='peakdetect', interpolate=10, lookahead=1)
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> fp = findpeaks(method='topology')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>> fp.plot_persistence()

        """
        if method is None: method='peakdetect'
        self.method = method
        self.type = 'peaks1d'
        if self.verbose>=3: print('[findpeaks] >Finding peaks in 1d-vector using [%s] method..' %(self.method))
        # Make numpy array
        X = np.array(X)
        Xraw = X.copy()
        result = {}

        # Interpolation
        if self.interpolate is not None:
            X = interpolate.interpolate_line1d(X, n=self.interpolate, method=2, showfig=False, verbose=self.verbose)

        # Compute peaks based on method
        if method=='peakdetect':
            # Peakdetect method
            max_peaks, min_peaks = peakdetect(X, lookahead=self.lookahead, delta=self.params['delta'])
            # Post processing for the peak-detect
            result['peakdetect'] = stats._post_processing(X, Xraw, min_peaks, max_peaks, self.interpolate, self.lookahead)
        elif method=='topology':
            # Compute persistence using toplogy method
            result = stats.topology(np.c_[X, X], limit=self.limit, verbose=self.verbose)
            # Post processing for the topology method
            result['topology'] = stats._post_processing(X, Xraw, result['valley'], result['peak'], self.interpolate, 1)
        elif method=='caerus':
            cs = caerus(**self.params)
            result = cs.fit(X, return_as_dict=True, verbose=self.verbose)
            # Post processing for the caerus method
            result['caerus'] = stats._post_processing(X, Xraw, np.c_[result['loc_start_best'], result['loc_start_best']], np.c_[result['loc_stop_best'], result['loc_stop_best']], self.interpolate, 1, labxRaw=result['df']['labx'].values)
            result['caerus']['model'] = cs
        else:
            if self.verbose>=2: print('[findpeaks] >WARNING: [method="%s"] is not supported in 1d-vector data. <return>' %(self.method))
            return None
        # Store
        self.results, self.args = self._store1d(X, Xraw, x, result)
        # Return
        return self.results

    # Store 1D vector
    def _store1d(self, X, Xraw, xs, result):
        # persist_score, res_peakd, results_topology
        # persist_score, results_peaksdetect, results_topology
        if xs is None: xs = np.arange(0, len(X))
        results = {}
        # Interpolated data
        dfint = pd.DataFrame()
        dfint['x'] = xs
        dfint['y'] = X
        # Store results for method
        if self.method=='peakdetect':
            # peakdetect
            dfint['labx'] = result['peakdetect']['labx_s']
            dfint['valley'] = False
            dfint['peak'] = False
            if result['peakdetect']['min_peaks_s'] is not None:
                dfint['valley'].iloc[result['peakdetect']['min_peaks_s'][:, 0].astype(int)] = True
            if result['peakdetect']['max_peaks_s'] is not None:
                dfint['peak'].iloc[result['peakdetect']['max_peaks_s'][:, 0].astype(int)] = True
        elif self.method=='topology':
            # Topology
            dfint['labx'] = result['topology']['labx_s']
            dfint['rank'] = result['Xranked']
            dfint['score'] = result['Xdetect']
            dfint['valley'] = False
            dfint['peak'] = False
            if result['topology']['min_peaks_s'] is not None:
                dfint['valley'].iloc[result['topology']['min_peaks_s'][:, 0].astype(int)] = True
            if result['topology']['max_peaks_s'] is not None:
                dfint['peak'].iloc[result['topology']['max_peaks_s'][:, 0].astype(int)] = True

            results['persistence'] = result['persistence']
            results['Xdetect'] = result['Xdetect']
            results['Xranked'] = result['Xranked']
            results['groups0'] = result['groups0']
        elif self.method=='caerus':
            # caerus
            dfint = result['df'].copy()
            dfint['y'] = result['X']
            dfint.drop(labels='X', inplace=True, axis=1)
            dfint['x'] = xs
            # dfint['labx'] = result['caerus']['labx_s']
            # dfint['valley'] = False
            # dfint['peak'] = False
            # if result['caerus']['min_peaks_s'] is not None:
            #     dfint['valley'].iloc[result['caerus']['min_peaks_s'][:, 0].astype(int)] = True
            # if result['caerus']['max_peaks_s'] is not None:
            #     dfint['peak'].iloc[result['caerus']['max_peaks_s'][:, 0].astype(int)] = True

        # As for the input data
        if self.interpolate is not None:
            df = pd.DataFrame()
            df['y'] = Xraw
            # Store results for method
            if self.method=='peakdetect':
                # peakdetect
                df['x'] = result['peakdetect']['xs']
                df['labx'] = result['peakdetect']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['peakdetect']['min_peaks'] is not None:
                    df['valley'].iloc[result['peakdetect']['min_peaks'][:, 0].astype(int)] = True
                if result['peakdetect']['max_peaks'] is not None:
                    df['peak'].iloc[result['peakdetect']['max_peaks'][:, 0].astype(int)] = True
            elif self.method=='topology':
                # Topology
                df['x'] = result['topology']['xs']
                df['labx'] = result['topology']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['topology']['min_peaks'] is not None:
                    df['valley'].iloc[result['topology']['min_peaks'][:, 0].astype(int)] = True
                if result['topology']['max_peaks'] is not None:
                    df['peak'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = True

                # Store the score and ranking
                df['rank'] = 0
                df['score'] = 0

                df['rank'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = dfint['rank'].iloc[result['topology']['max_peaks_s'][:, 0].astype(int)].values
                df['score'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = dfint['score'].iloc[result['topology']['max_peaks_s'][:, 0].astype(int)].values
                # df['rank'].loc[df['peak']] = dfint['rank'].loc[dfint['peak']].values
                # df['score'].loc[df['peak']] = dfint['score'].loc[dfint['peak']].values
            if self.method=='caerus':
                # caerus
                df['x'] = result['caerus']['xs']
                df['labx'] = result['df']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['caerus']['min_peaks'] is not None:
                    df['valley'].iloc[result['caerus']['min_peaks'][:, 0].astype(int)] = True
                if result['caerus']['max_peaks'] is not None:
                    df['peak'].iloc[result['caerus']['max_peaks'][:, 0].astype(int)] = True

            # Store in results
            results['df'] = df
            results['df_interp'] = dfint
        else:
            results['df'] = dfint

        if self.method=='caerus':
            results['model'] = result['caerus']['model']
        # Arguments
        args = {}
        args['method'] = self.method
        args['params'] = self.params
        args['lookahead'] = self.lookahead
        args['interpolate'] = self.interpolate
        args['figsize'] = self.figsize
        args['type'] = self.type
        # Return
        return results, args

    # Find peaks in 2D-array
    def peaks2d(self, X, method='topology'):
        """Detect peaks and valleys in a 2D-array or image.

        Description
        -----------
        To handle 2d-arrays or images. Use the .fit() function for more information regarding the input parameters:
            * method : method to be used for peak detection: 'topology', or 'mask'
            * limit : Values > limit are set as regions of interest (ROI).
            * scale : Scaling data in range [0-255] by img*(255/max(img))
            * denoise : Remove noise using method:
                * None
                * 'fastnl'
                * 'bilateral'
                * 'lee'
                * 'lee_enhanced'
                * 'lee_sigma'
                * 'kuan'
                * 'frost'
                * 'median'
                * 'mean'
            * window : Denoising window.
            * cu : noise variation coefficient
            * togray : Conversion to gray scale.
            * imsize : resize image
            * verbose : Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Parameters
        ----------
        X : array-like 1D vector.
            Input data.

        Returns
        -------
        dict()
            * Xraw    : The RAW input data
            * Xproc   : The pre-processed data
            * Xdetect : The detected peaks with the persistence scores (same shape as the input data)
            * XRanked : The detected peaks but based on the strenght (same shape as the input data)
            * persistence : pd.DataFrame()
                * x, y    : coordinates
                * birth   : Birth level
                * death   : Death level
                * score   : persistence scores

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
        >>> fp = findpeaks(denoise='fastnl', params={'window': 30}, imsize=(300,300))
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>>
        >>> # Plot each seperately
        >>> fp.plot_preprocessing()
        >>> fp.plot_persistence()
        >>> fp.plot_mesh()

        """
        if method is None: method='topology'
        self.method = method
        self.type = 'peaks2d'
        if self.verbose>=3: print('[findpeaks] >Finding peaks in 2d-array using %s method..' %(self.method))
        if (not self.togray) and (len(X.shape)==3) and (self.method=='topology'): raise Exception('[findpeaks] >Error: Topology method requires 2d-array. Your input is 3d. Hint: set togray=True.')

        # Preprocessing the image
        Xproc = self.preprocessing(X, showfig=False)
        # Compute peaks based on method
        if method=='topology':
            # Compute persistence based on topology method
            result = stats.topology2d(Xproc, limit=self.limit, whitelist=self.whitelist, verbose=self.verbose)
            # result = stats.topology(Xproc, limit=self.limit, verbose=self.verbose)
        elif method=='mask':
            # Compute peaks using local maximum filter.
            result = stats.mask(Xproc, limit=self.limit, verbose=self.verbose)
        else:
            if self.verbose>=2: print('[findpeaks] >WARNING: [method="%s"] is not supported in 2d-array (image) data. <return>' %(self.method))
            return None

        # Store
        self.results, self.args = self._store2d(X, Xproc, result)
        # Return
        if self.verbose>=3: print('[findpeaks] >Fin.')
        return self.results

    # Store 2D-array
    def _store2d(self, X, Xproc, result):
        # Store results
        results = {}
        results['Xraw'] = X
        results['Xproc'] = Xproc

        # Store method specific results
        if self.method=='topology':
            # results['topology'] = result
            results['Xdetect'] = result['Xdetect']
            results['Xranked'] = result['Xranked']
            results['persistence'] = result['persistence']
            # results['peak'] = result['peak'] # These values are incorrect when using 2d
            # results['valley'] = result['valley'] # These values are incorrect when using 2d
            results['groups0'] = result['groups0']
        if self.method=='mask':
            results['Xdetect'] = result['Xdetect']
            results['Xranked'] = result['Xranked']

        # Store arguments
        args = {}
        args['limit'] = self.limit
        args['scale'] = self.scale
        args['denoise'] = self.denoise
        args['togray'] = self.togray
        args['imsize'] = self.imsize
        args['figsize'] = self.figsize
        args['type'] = self.type
        # Return
        return results, args

    # Pre-processing
    def preprocessing(self, X, showfig=False):
        """Preprocessing steps of the 2D array (image).

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
        if showfig:
            # Number of axis to create:
            nplots = 1 + (self.imsize is not None) + self.scale + self.togray + (self.denoise is not None)
            fig, ax = plt.subplots(1, nplots, figsize=self.figsize)
            iax = 0

            # Plot RAW input image
            ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
            ax[iax].grid(False)
            ax[iax].set_title('Input\nRange: [%.3g,%.3g]' %(X.min(), X.max()))
            iax = iax + 1
            # plt.show()

        # Resize
        if self.imsize:
            X = stats.resize(X, size=self.imsize, verbose=self.verbose)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Resize\n(%s,%s)' %(self.imsize))
                iax = iax + 1
        # Scaling color range between [0,255]
        if self.scale:
            X = stats.scale(X, verbose=self.verbose)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Scale\nRange: [%.3g %.3g]' %(X.min(), X.max()))
                iax = iax + 1
        # Convert to gray image
        if self.togray:
            X = stats.togray(X, verbose=self.verbose)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Color conversion\nGray')
                iax = iax + 1
        # Denoising
        if self.denoise is not None:
            X = stats.denoise(X, method=self.denoise, window=self.window, cu=self.cu, verbose=self.verbose)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Denoise\n' + self.method)
                iax = iax + 1
        # Return
        return X

    # Pre-processing
    def imread(self, path, verbose=3):
        """Read file from disk or url.

        Parameters
        ----------
        path : String
            filepath or Url.

        Returns
        -------
        X : Numpy array

        """
        cv2 = stats._import_cv2()
        if is_url(path):
            if verbose>=3: print('[findpeaks] >Downloading from github source: [%s]' %(path))
            response = requests.get(path)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            X = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif os.path.isfile(path):
            if verbose>=3: print('[findpeaks] >Import [%s]' %(path))
            X = cv2.imread(path)
        # Return
        return X

    # %% Plotting
    def plot(self, limit=None, legend=True, figsize=None, cmap=None, text=True, s=None, marker='o', color='#000000',  xlabel='x-axis', ylabel='y-axis'):
        """Plot results.

        Parameters
        ----------
        legend : bool, (default: True)
            Show the legend.
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.
        cmap : object (default : None)
            Colormap. The default is derived wether image is convert to grey or not. Other options are: plt.cm.hot_r.
        text : Bool (default : True)
            Include text to the 2d-image that shows the peaks (p-number) and valleys (v-number)

        Returns
        -------
        fig_axis : tuple containing (fig, ax)

        """
        if not hasattr(self, 'results'):
            if self.verbose>=2: print('[findpeaks] >WARNING: Nothing to plot. <return>')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']

        if self.args['type']=='peaks1d':
            fig_axis = self.plot1d(legend=legend, figsize=figsize, xlabel=xlabel, ylabel=ylabel)
        elif self.args['type']=='peaks2d':
            # fig_axis = self.plot2d(figsize=figsize)
            fig_axis = self.plot_mask(figsize=figsize, cmap=cmap, text=text, limit=limit, s=s, marker=marker, color=color)
        else:
            if self.verbose>=2: print('[findpeaks] >WARNING: Nothing to plot for %s' %(self.args['type']))
            return None

        # Return
        return fig_axis

    def plot1d(self, legend=True, figsize=None, xlabel='x-axis', ylabel='y-axis'):
        """Plot the 1D results.

        Parameters
        ----------
        legend : bool, (default: True)
            Show the legend.
        figsize : (int, int), (default: None)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not self.args['type']=='peaks1d':
            if self.verbose>=3: print('[findpeaks] >Requires results of 1D data <return>.')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']
        ax1, ax2 = None, None
        title = self.method

        if self.method=='caerus':
            if self.results.get('model', None) is not None:
                ax = self.results['model'].plot(figsize=self.figsize)
                csplots._plot_graph(self.results['model'].results, figsize=self.figsize, xlabel=xlabel, ylabel=ylabel)
                # Return axis
                return ax
        else:
            # Make plot
            min_peaks, max_peaks = np.array([]), np.array([])
            df = self.results['df']
            if np.any('valley' in self.whitelist):
                min_peaks = df['x'].loc[df['valley']].values
            if np.any('peak' in self.whitelist):
                max_peaks = df['x'].loc[df['peak']].values
            ax1 = _plot_original(df['y'].values, df['x'].values, df['labx'].values, min_peaks.astype(int), max_peaks.astype(int), title=title, figsize=figsize, legend=legend, xlabel=xlabel, ylabel=ylabel)

            # Make interpolated plot
            if self.interpolate is not None:
                min_peaks, max_peaks = np.array([]), np.array([])
                df_interp = self.results['df_interp']
                if np.any('valley' in self.whitelist):
                    min_peaks = df_interp['x'].loc[df_interp['valley']].values
                if np.any('peak' in self.whitelist):
                    max_peaks = df_interp['x'].loc[df_interp['peak']].values
                ax2 = _plot_original(df_interp['y'].values, df_interp['x'].values, df_interp['labx'].values, min_peaks.astype(int), max_peaks.astype(int), title=title + ' (interpolated)', figsize=figsize, legend=legend, xlabel=xlabel, ylabel=ylabel)
            # Return axis
            return (ax2, ax1)

    def plot2d(self, figsize=None, limit=None):
        """Plot the 2d results.

        Parameters
        ----------
        figsize : (int, int), (default: None)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not self.args['type']=='peaks2d':
            if self.verbose>=3: print('[findpeaks] >Requires results of 2D data <return>.')
            return None
        ax_method, ax_mesh = None, None
        figsize = figsize if figsize is not None else self.args['figsize']
        # Plot preprocessing steps
        self.plot_preprocessing()

        # Setup figure
        if self.method=='mask':
            ax_method = self.plot_mask(figsize=figsize, limit=limit)
        if self.method=='topology':
            # Plot topology/persistence
            ax_method = self.plot_persistence(figsize=figsize)

        # Plot mesh
        ax_mesh = self.plot_mesh(figsize=figsize)

        # Return axis
        return (ax_method, ax_mesh)

    def plot_preprocessing(self):
        """Plot the pre-processing steps.

        Returns
        -------
        None.

        """
        if (not hasattr(self, 'results')) or (self.type=='peaks1d'):
            if self.verbose>=2: print('[findpeaks] >WARNING: Nothing to plot. Hint: run fit(X), where X is the (image) data. <return>')
            return None

        _ = self.preprocessing(X=self.results['Xraw'], showfig=True)

    def plot_mask(self, limit=None, figsize=None, cmap=None, text=True, s=10, marker='o', color='#000000'):
        """Plot the masking.

        Parameters
        ----------
        limit : float, (default : None)
            Values > limit are set as regions of interest (ROI).
        figsize : (int, int), (default: None)
            (width, height) in inches.
        cmap : object (default : None)
            Colormap. The default is derived wether image is convert to grey or not. Other options are: plt.cm.hot_r.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if (self.type=='peaks1d'):
            if self.verbose>=2: print('[findpeaks] >WARNING: Nothing to plot. Hint: run fit(X), where X is the 2d-array (image). <return>')
            return None

        if limit is None: limit = self.limit
        # Only show above the limit
        Xdetect = self.results['Xdetect'].copy()
        if limit is not None:
            Xdetect[np.abs(Xdetect)<limit]=0
        if cmap is None:
            cmap = 'gray' if self.args['togray'] else None
            cmap = cmap + '_r'
        # Get the index for the detected peaks/valleys
        idx_peaks = np.where(Xdetect>0)
        idx_valleys = np.where(Xdetect<0)

        # Setup figure
        figsize = figsize if figsize is not None else self.args['figsize']
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Plot input image
        ax1.imshow(self.results['Xraw'], cmap, interpolation="nearest")
        ax1.set_title('Input')
        ax1.grid(False)

        # For vizualisation purposes, plot all absolute numbers
        Xproc = self.results['Xproc'].copy()
        Xproc[idx_peaks]=0
        Xproc[idx_valleys]=1
        ax2.imshow(Xproc, cmap, interpolation="nearest")
        ax2.set_title('Processed image')
        ax2.grid(False)

        # Masking
        ax3.imshow(np.abs(Xdetect), 'gray_r', interpolation="nearest")
        ax3.set_title(self.method + ' (' + str(len(np.where(Xdetect>0)[0])) + ' peaks and ' + str(len(np.where(Xdetect<0)[0])) + ' valleys)')
        ax3.grid(False)

        X = self.results['persistence'].loc[self.results['persistence']['score']>limit,:]
        for i in range(X.shape[0]):
            if s is None:
                X['score'] = stats.normalize(X['score'].values, minscale=2, maxscale=10, scaler='minmax')
            else:
                X['score'] = s
            ax1.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)
            ax2.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)
            ax3.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)

        if text:
            for idx in tqdm(zip(idx_peaks[0], idx_peaks[1]), disable=disable_tqdm(self.verbose)):
                ax2.text(idx[1], idx[0], 'p' + self.results['Xranked'][idx].astype(str))
                ax3.text(idx[1], idx[0], 'p' + self.results['Xranked'][idx].astype(str))

            for idx in tqdm(zip(idx_valleys[0], idx_valleys[1]), disable=disable_tqdm(self.verbose)):
                ax2.text(idx[1], idx[0], 'v' + self.results['Xranked'][idx].astype(str))
                ax3.text(idx[1], idx[0], 'v' + self.results['Xranked'][idx].astype(str))

        # Show plot
        plt.show()
        # Return
        return (ax1, ax2, ax3)

    def plot_mesh(self,
                  wireframe=True,
                  surface=True,
                  rstride=2,
                  cstride=2,
                  cmap=plt.cm.hot_r,
                  view=None,
                  xlim=None,
                  ylim=None,
                  zlim=None,
                  title='',
                  figsize=None,
                  savepath=None):
        """Plot the 3d-mesh.

        Parameters
        ----------
        wireframe : bool, (default is True)
            Plot the wireframe
        surface : bool, (default is True)
            Plot the surface
        rstride : int, (default is 2)
            Array row stride (step size).
        cstride : int, (default is 2)
            Array column stride (step size).
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.
        view : tuple, (default : None)
            * Rotate the mesh plot.
            * (0, 0) : y vs z
            * (0, 90) : x vs z
            * (90, 0) : y vs x
            * (90, 90) : x vs y
        cmap : object
            Colormap. The default is plt.cm.hot_r.
        xlim : tuple(int, int), (default: None)
            x-limit in the axis.
            None: No limit.
            [1, 5]: Limit between the range 1 and 5.
            [1, None]: Limit between range 1 and unlimited.
            [None, 5]: Limit between range unlimited and 5.
        ylim : tuple(int, int), (default: None)
            y-limit in the axis.
            None: No limit.
            [1, 5]: Limit between the range 1 and 5.
            [1, None]: Limit between range 1 and unlimited.
            [None, 5]: Limit between range unlimited and 5.
        zlim : tuple(int, int), (default: None)
            z-limit in the axis.
            None: No limit.
            [1, 5]: Limit between the range 1 and 5.
            [1, None]: Limit between range 1 and unlimited.
            [None, 5]: Limit between range unlimited and 5.
        figsize : (int, int), (default: None)
            (width, height) in inches.
        savepath : bool (default : None)
            Path with filename to save the figure, eg: './tmp/my_image.png'
        verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.
        
        Example
        -------
        >>> # Import library
        >>> from findpeaks import findpeaks
        >>> #
        >>> # Initialize
        >>> fp = findpeaks(method='topology', scale=False, denoise=None, togray=False, imsize=False, params={'window': 15})
        >>> #
        >>> # Load example data set
        >>> X = fp.import_example('2dpeaks')
        >>> #
        >>> # Fit model
        >>> fp.fit(X)
        >>> #
        >>> # Create mesh plot
        >>> fp.plot_mesh()
        >>> # Create mesh plot with limit on x-axis and y-axis
        >>> fp.plot_mesh(xlim=[10, 30], ylim=[4, 10], zlim=[None, 8])

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not hasattr(self, 'results'):
            if self.verbose>=2: print('[findpeaks] >WARNING: Nothing to plot. Hint: run the fit() function. <return>')
            return None
        if self.results.get('Xproc', None) is None:
            if self.verbose>=3: print('[findpeaks] >These analysis do not support mesh plotting. This may be caused because your are analysing 1D.')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']
        if self.verbose>=3: print('[findpeaks] >Plotting 3d-mesh..')
        ax1, ax2 = None, None
        if savepath is not None:
            savepath = str.replace(savepath, ',', '_')
            savepath = str.replace(savepath, '=', '_')

        # Compute meshgrid
        Z = self.results['Xproc'].copy()
        X, Y = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
        # To limit on the X and Y axis, we need to create a trick by setting all values to nan in the Z-axis that should be limited.
        if xlim is not None:
            if xlim[0] is not None: Z[X<xlim[0]]=np.nan
            if xlim[1] is not None: Z[X>xlim[1]]=np.nan
        if ylim is not None:
            if ylim[0] is not None: Z[Y<ylim[0]]=np.nan
            if ylim[1] is not None: Z[Y>ylim[1]]=np.nan
        if zlim is not None:
            if zlim[0] is not None: Z[Z<zlim[0]]=np.nan
            if zlim[1] is not None: Z[Z>zlim[1]]=np.nan

        # Plot the figure
        if wireframe:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(projection='3d')
            ax1 = fig.gca()
            ax1.plot_wireframe(X, Y, Z, rstride=rstride, cstride=cstride, linewidth=0.8)
            ax1.set_xlabel('x-axis')
            ax1.set_ylabel('y-axis')
            ax1.set_zlabel('z-axis')
            if view is not None:
                ax1.view_init(view[0], view[1])
                # ax1.view_init(50, -10) # x vs y
            ax1.set_title(title)
            if xlim is not None: ax1.set_xlim3d(xlim[0], xlim[1])
            if ylim is not None: ax1.set_ylim3d(ylim[0], ylim[1])
            if zlim is not None: ax1.set_zlim3d(zlim[0], zlim[1])

            plt.show()
            if savepath is not None:
                if self.verbose>=3: print('[findpeaks] >Saving wireframe to disk..')
                fig.savefig(savepath)

        if surface:
            # Plot the figure
            fig = plt.figure(figsize=figsize)
            ax2 = fig.add_subplot(projection='3d')
            ax2 = fig.gca()
            ax2.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, cmap=cmap, linewidth=0, shade=True, antialiased=False)
            if view is not None:
                ax2.view_init(view[0], view[1])
            ax2.set_xlabel('x-axis')
            ax2.set_ylabel('y-axis')
            ax2.set_zlabel('z-axis')
            ax2.set_title(title)
            if xlim is not None: ax2.set_xlim3d(xlim[0], xlim[1])
            if ylim is not None: ax2.set_ylim3d(ylim[0], ylim[1])
            if zlim is not None: ax2.set_zlim3d(zlim[0], zlim[1])
            plt.show()
            if savepath is not None:
                if self.verbose>=3: print('[findpeaks] >Saving surface to disk..')
                fig.savefig(savepath)

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

    def plot_persistence(self, figsize=(20, 8), fontsize_ax1=14, fontsize_ax2=14, xlabel='x-axis', ylabel='y-axis', verbose=None):
        """Plot the homology-peristence.

        Parameters
        ----------
        figsize : (int, int), (default: None)
            (width, height) in inches.
        fontsize_ax1 : int, (default: 14)
            Font size for the labels in the left figure. Choose None for no text-labels.
        fontsize_ax2 : int, (default: 14)
            Font size for the labels in the right figure. Choose None for no text-labels.
        verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        Returns
        -------
        ax1 : object
            Figure axis 1.
        ax2 : object
            Figure axis 2.

        """
        if verbose is None: verbose=self.verbose
        if (self.method!='topology') or (not hasattr(self, 'results')) or len(self.results['persistence']['birth_level'].values)<=0:
            if verbose>=3: print('[findpeaks] >WARNING: Nothing to plot. Hint: run the .fit(method="topology") function. <return>')
            return None

        # Setup figure
        figsize = figsize if figsize is not None else self.args['figsize']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Create the persistence ax2
        ax2 = self._plot_persistence_ax2(fontsize_ax2, ax2, verbose)
        # Create the persistence ax1
        ax1, ax2 = self._plot_persistence_ax1(fontsize_ax1, ax1, ax2, figsize, xlabel, ylabel, verbose)
        # Plot
        plt.show()
        # Return
        return ax1, ax2

    def _plot_persistence_ax1(self, fontsize, ax1, ax2, figsize, xlabel, ylabel, verbose):
        if self.args['type']=='peaks1d':
            # Attach the ranking-labels
            if fontsize is not None:
                y = self.results['df']['y'].values
                x = self.results['df']['x'].values
                idx = np.where(self.results['df']['rank']>0)[0]
                for i in tqdm(idx, disable=disable_tqdm(verbose)):
                    ax1.text(x[i], (y[i] + y[i] * 0.01), str(self.results['df']['rank'].iloc[i]), color='b', fontsize=fontsize)

            # minpers = 0
            min_peaks, max_peaks = np.array([]), np.array([])
            if np.any('valley' in self.whitelist):
                min_peaks = self.results['df']['x'].loc[self.results['df']['valley']].values
            if np.any('peak' in self.whitelist):
                max_peaks = self.results['df']['x'].loc[self.results['df']['peak']].values
            # Make the plot
            ax1 = _plot_original(self.results['df']['y'].values, self.results['df']['x'].values, self.results['df']['labx'].values, min_peaks.astype(int), max_peaks.astype(int), title='Persistence', figsize=figsize, legend=True, ax=ax1, xlabel=xlabel, ylabel=ylabel)
            # Set limits
            X = np.c_[self.results['df']['x'].values, self.results['df']['y'].values]
            ax1.set_xlim((np.min(X), np.max(X)))
            ax1.set_ylim((np.min(X), np.max(X)))
        else:
            # X = self.results['Xproc']
            # Make the figure
            Xdetect = np.zeros_like(self.results['Xproc']).astype(int)
            # fig, ax1 = plt.subplots()
            # minpers = 1
            # Plot the detected loci
            if verbose>=3: print('[findpeaks] >Plotting loci of birth..')
            ax1.set_title("Loci of births")
            for i, homclass in tqdm(enumerate(self.results['groups0']), disable=disable_tqdm(verbose)):
                p_birth, bl, pers, p_death = homclass
                if (self.limit is None):
                    y, x = p_birth
                    Xdetect[y, x] = i + 1
                    ax1.plot([x], [y], '.', c='b')
                    ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize)
                elif pers > self.limit:
                    y, x = p_birth
                    Xdetect[y, x] = i + 1
                    ax1.plot([x], [y], '.', c='b')
                    ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize)

            ax1.set_xlim((0, self.results['Xproc'].shape[1]))
            ax1.set_ylim((0, self.results['Xproc'].shape[0]))
            ax1.invert_yaxis()
            plt.gca().invert_yaxis()
            ax1.grid(True)
            # Set the axis to 255-255 in ax2 because it is an image.
            ax2.plot([0, 255], [0, 255], '-', c='grey')
        return ax1, ax2

    def _plot_persistence_ax2(self, fontsize, ax2, verbose=3):
        x = self.results['persistence']['birth_level'].values
        y = self.results['persistence']['death_level'].values
        if len(x)<=0:
            if verbose>=3: print('[[findpeaks]> Nothing to plot.')
            return None
        ax2.plot(x, y, '.', c='b')
        if fontsize is not None:
            for i in tqdm(range(0, len(x)), disable=disable_tqdm(verbose)):
                ax2.text(x[i], (y[i] + y[i] * 0.01), str(i + 1), color='b', fontsize=fontsize)

        X = np.c_[x, y]
        ax2.plot([np.min(X), np.max(X)], [np.min(X), np.max(X)], '-', c='grey')
        ax2.set_xlabel("Birth level")
        ax2.set_ylabel("Death level")
        ax2.set_xlim((np.min(X), np.max(X)))
        ax2.set_ylim((np.min(X), np.max(X)))
        ax2.grid(True)
        return ax2

    def import_example(self, data='2dpeaks', url=None, sep=';', datadir=None):
        """Import example dataset from github source.

        Description
        -----------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: "1dpeaks", "2dpeaks", "2dpeaks_image", 'btc', 'facebook'
        url : str
            url link to to dataset.
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.
        datadir : path-like
            Directory to store downloaded datasets in. Defaults to data sub-directory
            of findpeaks install location.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        X = import_example(data=data, url=url, sep=sep, verbose=self.verbose, datadir=datadir)
        return X


# %%
def _plot_original(X, xs, labx, min_peaks, max_peaks, title=None, legend=True, ax=None, figsize=(15, 8), xlabel=None, ylabel=None):
    uilabx = np.unique(labx)
    uilabx = uilabx[~np.isnan(uilabx)]

    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, X, 'k')
    if np.any(max_peaks):
        ax.plot(max_peaks, X[max_peaks], "x", label='Peak')
    if np.any(min_peaks):
        ax.plot(min_peaks, X[min_peaks], "o", label='Valley')

    # Color each detected label
    s=np.arange(0, len(X))
    for i in uilabx:
        idx=(labx==i)
        ax.plot(s[idx], X[idx])
        # plt.plot(s[idx], X[idx], label='peak' + str(i))

    if legend: ax.legend(loc=0)
    ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()
    return ax


# %% Import example dataset from github.
def import_example(data='2dpeaks', url=None, sep=';', verbose=3, datadir=None):
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
    datadir : path-like
        Directory to store downloaded datasets in. Defaults to data sub-directory
        of findpeaks install location.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    if url is not None:
        fn = os.path.basename(urlparse(url).path).strip()
        if not fn:
            if verbose>=3: print('[findpeaks] >Could not determine filename to download <return>.')
            return None
        data, _ = os.path.splitext(fn)
    elif data=='2dpeaks_image':
        url='https://erdogant.github.io/datasets/' + data + '.png'
        fn = "2dpeaks_image.png"
    elif data=='2dpeaks':
        url='https://erdogant.github.io/datasets/' + data + '.zip'
        fn = "2dpeaks.zip"
    elif data=='1dpeaks':
        # x = [0,   13,  22,  30,  35,  38,   42,   51,   57,   67,  73,   75,  89,   126,  141,  150,  200 ]
        y = [1.5, 0.8, 1.2, 0.2, 0.4, 0.39, 0.42, 0.22, 0.23, 0.1, 0.11, 0.1, 0.14, 0.09, 0.04, 0.02, 0.01]
        # X = np.c_[x, y]
        return y
    elif (data=='btc') or (data=='facebook'):
        from caerus import caerus
        cs = caerus()
        X = cs.download_example(name=data, verbose=verbose)
        return X
    else:
        if verbose>=2: print('[findpeaks] >WARNING: Nothing to download. <return>')
        return None

    if datadir is None: datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(datadir, fn)
    if not os.path.isdir(datadir):
        os.makedirs(datadir, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[findpeaks] >Downloading from github source: [%s]' %(url))
        r = requests.get(url, stream=True)
        with open(PATH_TO_DATA, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    # Import local dataset
    if verbose>=3: print('[findpeaks] >Import [%s]' %(PATH_TO_DATA))
    if data=='2dpeaks_image':
        cv2 = stats._import_cv2()
        X = cv2.imread(PATH_TO_DATA)
    else:
        X = pd.read_csv(PATH_TO_DATA, sep=sep).values
    # Return
    return X


# Check url
def is_url(url):
    try:
        _ = urlparse(url)
        return True
    except ValueError:
        return False

# %%
# def disable_tqdm(verbose):
    # """Set the verbosity messages."""
    # return  (True if ((verbose==0 or verbose is None) or verbose>3) else False)
