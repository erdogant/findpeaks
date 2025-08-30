"""Python library for the detection of peaks and valleys."""
# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : See LICENSE
# ----------------------------------------------------

# import findpeaks
from caerus import caerus
import caerus.utils.csplots as csplots
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import requests
from urllib.parse import urlparse
import logging
from adjustText import adjust_text

try:
    from findpeaks.peakdetect import peakdetect
    import findpeaks.stats as stats
    from findpeaks.stats import disable_tqdm
    import findpeaks.interpolate as interpolate
except:
    #### DEBUG ONLY ####
    from peakdetect import peakdetect
    import stats as stats
    from stats import disable_tqdm
    import interpolate as interpolate
# #####################

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
   logging.basicConfig(level=logging.INFO, format='[{asctime}] [{name}] [{levelname}] {msg}', style='{', datefmt='%d-%m-%Y %H:%M:%S')


# %%
class findpeaks():
    """Python library for robust detection and analysis of peaks and valleys in 1D and 2D data.

    findpeaks is a comprehensive library for detecting, analyzing, and visualizing peaks and valleys 
    in both 1D vectors and 2D arrays (images). It provides multiple detection methods including 
    topology-based persistent homology, traditional peak detection algorithms, and mask-based approaches.
    
    **Key Features:**
    - **Multiple Detection Methods**: Topology (persistent homology), peakdetect, caerus, and mask-based detection
    - **1D and 2D Support**: Works with time series, signals, images, and spatial data
    - **Advanced Preprocessing**: Denoising, scaling, interpolation, and image preprocessing
    - **Rich Visualization**: Persistence diagrams, 3D mesh plots, preprocessing steps, and masking plots
    - **Robust Analysis**: Mathematically stable methods with quantitative significance measures
    
    **Detection Methods:**
    - **Topology**: Uses persistent homology for robust peak detection with mathematical stability
    - **Peakdetect**: Traditional algorithm for 1D peak detection with lookahead and delta parameters
    - **Caerus**: Advanced peak detection with multiple optimization strategies
    - **Mask**: Local maximum filtering for 2D peak detection
    
    **Applications:**
    - Signal processing and time series analysis
    - Image processing and feature detection
    - Hough Transform applications with enhanced robustness
    - Scientific data analysis and visualization
    - Peak finding in noisy or complex datasets

    Examples
    --------
    >>> from findpeaks import findpeaks
    >>> 
    >>> # 1D vector example
    >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
    >>> fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10)
    >>> results = fp.fit(X)
    >>> fp.plot()
    >>>
    >>> # 2D array example
    >>> X = fp.import_example('2dpeaks')
    >>> results = fp.fit(X)
    >>> fp.plot()
    >>>
    >>> # Image processing example
    >>> fp = findpeaks(method='topology', imsize=(300,300), denoise='fastnl', params={'window': 30})
    >>> X = fp.import_example('2dpeaks_image')
    >>> results = fp.fit(X)
    >>> fp.plot()
    >>>
    >>> # Advanced visualization
    >>> fp.plot_preprocessing()  # Show preprocessing steps
    >>> fp.plot_persistence()    # Show persistence diagram
    >>> fp.plot_mesh()          # Show 3D mesh plot

    References
    ----------
    * https://erdogant.github.io/findpeaks/
    * Johannes Ferner et al, Persistence-based Hough Transform for Line Detection, https://arxiv.org/abs/2504.16114

    """
    def __init__(self,
                 method=None,
                 whitelist=['peak', 'valley'],
                 lookahead=200,
                 interpolate=None,
                 limit=None,
                 imsize=None,
                 scale=True,
                 togray=True,
                 denoise='fastnl',
                 window=None,  # DEPRECATED IN LATER VERSIONS: specify in params
                 cu=None,  # DEPRECATED IN LATER VERSIONS: specify in params
                 params_caerus={},  # DEPRECATED IN LATER VERSIONS: use params instead
                 params={'window': 3, 'delta': 0},
                 figsize=(15, 8),
                 verbose='info'):
        """Initialize findpeaks with detection and preprocessing parameters.

        Parameters
        ----------
        method : str, optional (default: None)
            Peak detection method to use. If None, defaults are chosen based on data type.
            
            **1D-vector methods:**
            - 'topology': Persistent homology-based detection (most robust)
            - 'peakdetect': Traditional algorithm with lookahead and delta parameters
            - 'caerus': Advanced peak detection with optimization strategies
            
            **2D-array methods:**
            - 'topology': Persistent homology-based detection (default, most robust)
            - 'mask': Local maximum filtering for peak detection
            
        whitelist : str or list, optional (default: ['peak', 'valley'])
            Types of features to detect:
            - 'peak': Detect only peaks (local maxima)
            - 'valley': Detect only valleys (local minima)
            - ['peak', 'valley']: Detect both peaks and valleys
            
        lookahead : int, optional (default: 200)
            Number of points to look ahead when detecting peaks (peakdetect method).
            For small datasets (< 50 points), use values of 1-2.
            Higher values provide more robust detection but may miss closely spaced peaks.
            
        interpolate : int, optional (default: None)
            Interpolation factor for 1D data smoothing. Higher values create smoother curves
            with less sharp edges. Useful for noisy data preprocessing.
            
        limit : float, optional (default: None)
            Persistence threshold for topology method. Only peaks with persistence > limit
            are considered significant. Lower values detect more peaks, higher values
            detect only the most prominent peaks.
            
        imsize : tuple, optional (default: None)
            Target image size for 2D data preprocessing: (width, height).
            Useful for reducing computation time on large images.
            
        scale : bool, optional (default: True)
            Scale image values to range [0-255] for consistent processing.
            
        togray : bool, optional (default: True)
            Convert color images to grayscale. Required for topology method.
            
        denoise : str, optional (default: 'fastnl')
            Denoising method for 2D data preprocessing:
            - None: No denoising
            - 'fastnl': Fast Non-Local Means (recommended)
            - 'bilateral': Bilateral filtering (preserves edges)
            - 'lee': Lee filter for SAR images
            - 'lee_enhanced': Enhanced Lee filter
            - 'lee_sigma': Lee filter with sigma parameter
            - 'kuan': Kuan filter for SAR images
            - 'frost': Frost filter for SAR images
            - 'median': Median filtering
            - 'mean': Mean filtering
            
        params : dict, optional (default: {'window': 3, 'delta': 0})
            Method-specific parameters:
            
            **For caerus method:**
            - 'window': int (default: 50) - Window size for analysis
            - 'minperc': int (default: 3) - Minimum percentage threshold
            - 'nlargest': int (default: 10) - Number of largest peaks to return
            - 'threshold': float (default: 0.25) - Detection threshold
            
            **For lee_sigma method:**
            - 'window': int (default: 7) - Window size
            - 'sigma': float (default: 0.9) - Speckle noise standard deviation
            - 'num_looks': int (default: 1) - Number of looks in SAR image
            - 'tk': int (default: 5) - Threshold for neighboring pixels
            
            **For peakdetect method:**
            - 'delta': int (default: 0) - Minimum difference between peak and following points
              Set to >= RMSnoise * 5 for optimal performance
              
            **For denoising methods:**
            - 'window': int (default: 3) - Denoising window size
            - 'cu': float (default: 0.25) - Noise variation coefficient (kuan, lee, lee_enhanced)
            
        figsize : tuple, optional (default: (15, 8))
            Default figure size for plots: (width, height) in inches.
            
        verbose : str or int, optional (default: 'info')
            Logging verbosity level:
            - 'silent', 'off', 'no': No messages
            - 'critical': Critical errors only
            - 'error': Errors and critical messages
            - 'warning': Warnings and above
            - 'info': Information messages and above (default)
            - 'debug': All messages including debug information

        Returns
        -------
        findpeaks
            Initialized findpeaks object ready for peak detection.

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # Basic 1D peak detection
        >>> fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10)
        >>> 
        >>> # Advanced 2D detection with preprocessing
        >>> fp = findpeaks(method='topology', imsize=(300,300), denoise='fastnl', 
        ...                params={'window': 30}, limit=0.1)
        >>> 
        >>> # Custom parameters for specific use case
        >>> fp = findpeaks(method='caerus', params={'window': 50, 'minperc': 5})

        Notes
        -----
        - The topology method provides the most robust detection with mathematical stability
        - For noisy data, use denoising preprocessing before detection
        - The limit parameter is crucial for topology method to filter significant peaks
        - Interpolation can help with noisy 1D data but may smooth out important peaks

        References
        ----------
        * https://erdogant.github.io/findpeaks/
        * Johannes Ferner et al, Persistence-based Hough Transform for Line Detection, https://arxiv.org/abs/2504.16114
        """
        if window is not None: logger.info(
            'The input parameter "window" will be deprecated in future releases. Please use "params={"window": 5}" '
            'instead.')
        if cu is not None: logger.info(
            'The input parameter "cu" will be deprecated in future releases. Please use "params={"cu": 3}" instead.')

        # Set the logger
        set_logger(verbose=verbose)

        # Store in object
        if isinstance(whitelist, str): whitelist = [whitelist]
        if lookahead is None: lookahead = 1
        lookahead = np.maximum(1, lookahead)
        # if method is None: raise Exception('Specify the desired method="topology", "peakdetect",
        # or "mask".')
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
        # self.height = height

        # Store parameters for caerus
        defaults = {}
        if method == 'caerus':
            if len(params_caerus) > 0:
                logger.info('The input parameter "params_caerus" will be deprecated in future releases. Please use "params" instead.')
                params = params_caerus
            defaults = {'window': 50, 'minperc': 3, 'nlargest': 10, 'threshold': 0.25}
        elif method == 'lee_sigma':
            defaults = {'window': 7, 'sigma': 0.9, 'num_looks': 1, 'tk': 5}
        elif method == 'peakdetect':
            defaults = {'delta': 0}
        defaults = {**{'window': 3}, **defaults}

        params = {**defaults, **params}
        self.window = params['window']
        self.cu = params.get('cu', 0.25)
        self.params = params

    def fit(self, X, x=None):
        """Detect peaks and valleys in 1D vector or 2D-array data.

        Description
        -----------
        Performs peak and valley detection on the input data using the configured method.
        Automatically determines whether the input is 1D or 2D and applies appropriate
        preprocessing and detection algorithms.

        Parameters
        ----------
        X : array-like
            Input data for peak detection. Can be:
            - 1D array/list: Time series, signal data, or vector data
            - 2D array: Image data, spatial data, or matrix data
            - pandas DataFrame: Will be converted to numpy array
            
        x : array-like, optional (default: None)
            X-coordinates for 1D data. If None, uses sequential indices [0, 1, 2, ...].
            Only used for 1D data visualization and result mapping.

        Returns
        -------
        dict
            Dictionary containing detection results with the following keys:
            
            **For 1D data:**
            - 'df': pandas DataFrame with original data coordinates and detection results
            - 'df_interp': pandas DataFrame with interpolated data (if interpolation used)
            - 'persistence': pandas DataFrame with persistence scores (topology method only)
            - 'Xdetect': numpy array with detection scores
            - 'Xranked': numpy array with ranked peak/valley indices
            
            **For 2D data:**
            - 'Xraw': numpy array of original input data
            - 'Xproc': numpy array of preprocessed data
            - 'Xdetect': numpy array with detection scores (same shape as input)
            - 'Xranked': numpy array with ranked peak/valley indices (same shape as input)
            - 'persistence': pandas DataFrame with persistence scores (topology method only)
            - 'groups0': list of homology groups (topology method only)
            
            **DataFrame columns (when applicable):**
            - 'x', 'y': Coordinates of detected features
            - 'peak': Boolean indicating if point is a peak
            - 'valley': Boolean indicating if point is a valley
            - 'score': Persistence or detection score
            - 'rank': Ranking of feature significance (1 = most significant)
            - 'labx': Label/group assignment for connected components

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # 1D peak detection
        >>> X = [1, 2, 5, 3, 2, 1, 4, 6, 4, 2, 1]
        >>> fp = findpeaks(method='peakdetect', lookahead=1)
        >>> results = fp.fit(X)
        >>> print(f"Found {results['df']['peak'].sum()} peaks")
        >>> 
        >>> # 2D peak detection
        >>> X = fp.import_example('2dpeaks')
        >>> fp = findpeaks(method='topology', limit=0.1)
        >>> results = fp.fit(X)
        >>> print(f"Found {len(results['persistence'])} significant features")

        Notes
        -----
        - The method automatically handles data type conversion and preprocessing
        - Results are stored in the object for subsequent plotting and analysis
        - For topology method, persistence scores provide quantitative significance measures
        - Interpolation (if enabled) creates smoother data for better detection

        See Also
        --------
        peaks1d : 1D peak detection method
        peaks2d : 2D peak detection method
        plot : Visualize detection results
        """
        # Check datatype
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, type(pd.DataFrame())):
            X = X.values

        if len(X.shape) > 1:
            # 2d-array (image)
            results = self.peaks2d(X, method=self.method)
        else:
            # 1d-array (vector)
            results = self.peaks1d(X, x=x, method=self.method)

        return results

    # Find peaks in 1D vector
    def peaks1d(self, X, x=None, method='peakdetect', height=0):
        """Detect peaks and valleys in 1D array data.

        Method Description
        ------------------
        Performs peak and valley detection on 1D data using the specified method.
        Supports multiple detection algorithms with different characteristics:
        
        - **topology**: Persistent homology-based detection (most robust, handles noise well)
        - **peakdetect**: Traditional algorithm with lookahead and delta parameters
        - **caerus**: Advanced peak detection with optimization strategies

        Parameters
        ----------
        X : array-like
            1D input data (vector, time series, or signal data)
            
        x : array-like, optional (default: None)
            X-coordinates for the data points. If None, uses sequential indices.
            
        method : str, optional (default: 'peakdetect')
            Detection method to use:
            - 'topology': Persistent homology-based detection
            - 'peakdetect': Traditional peak detection algorithm
            - 'caerus': Advanced peak detection with optimization
            
        height : float, optional (default: 0)
            Minimum height requirement for peaks (peakdetect method only)

        Returns
        -------
        dict
            Dictionary containing detection results:
            
            **Common keys:**
            - 'df': pandas DataFrame with original data and detection results
            - 'df_interp': pandas DataFrame with interpolated data (if interpolation used)
            
            **For topology method:**
            - 'persistence': pandas DataFrame with persistence scores and coordinates
            - 'Xdetect': numpy array with detection scores
            - 'Xranked': numpy array with ranked peak/valley indices
            - 'groups0': list of homology groups
            
            **For peakdetect method:**
            - 'peakdetect': dict with peak detection results
            
            **For caerus method:**
            - 'caerus': dict with caerus detection results
            - 'model': caerus model object
            
            **DataFrame columns:**
            - 'x', 'y': Coordinates and values
            - 'peak': Boolean indicating peak locations
            - 'valley': Boolean indicating valley locations
            - 'score': Detection or persistence scores
            - 'rank': Feature ranking (1 = most significant)
            - 'labx': Component labels

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # Basic peak detection
        >>> X = [1, 2, 5, 3, 2, 1, 4, 6, 4, 2, 1]
        >>> fp = findpeaks(method='peakdetect', lookahead=1)
        >>> results = fp.peaks1d(X)
        >>> 
        >>> # Topology-based detection with persistence
        >>> fp = findpeaks(method='topology', limit=0.1)
        >>> results = fp.peaks1d(X)
        >>> print(f"Persistence scores: {results['persistence']['score'].values}")

        Notes
        -----
        - The topology method provides the most robust detection with mathematical stability
        - Interpolation (if enabled) creates smoother data for better detection
        - Persistence scores quantify the significance of detected features
        - Results are automatically stored in the object for plotting and analysis

        """
        if method is None: method = 'peakdetect'
        # if x is not None: x = x.astype(float)
        # X = X.astype(float)
        self.method = method
        self.type = 'peaks1d'
        logger.debug('Finding peaks in 1d-vector using [%s] method..' % (self.method))
        # Make numpy array
        X = np.array(X)
        Xraw = X.copy()
        result = {}

        # Interpolation
        if self.interpolate is not None and self.interpolate>0:
            X = interpolate.interpolate_line1d(X, n=self.interpolate, method=2, showfig=False)

        # Compute peaks based on method
        if method == 'peakdetect':
            # Peakdetect method
            # max_peaks, min_peaks = peakdetect(X, lookahead=self.lookahead, delta=self.params['delta'], height=self.height)
            max_peaks, min_peaks = peakdetect(X, lookahead=self.lookahead, delta=self.params['delta'])
            # Post processing for the peak-detect
            result['peakdetect'] = stats._post_processing(X, Xraw, min_peaks, max_peaks, self.interpolate,
                                                          self.lookahead)
        elif method == 'topology':
            # Compute persistence using toplogy method
            result = stats.topology(np.c_[X, X], limit=self.limit)
            # Post processing for the topology method
            result['topology'] = stats._post_processing(X, Xraw, result['valley'], result['peak'], self.interpolate, 1)
        elif method == 'caerus':
            caerus_params = self.params.copy()
            if caerus_params.get('delta') is not None: caerus_params.pop('delta')
            cs = caerus(**caerus_params)
            result = cs.fit(X, return_as_dict=True)
            # Post processing for the caerus method
            result['caerus'] = stats._post_processing(X, Xraw,
                                                      np.c_[result['loc_start_best'], result['loc_start_best']],
                                                      np.c_[result['loc_stop_best'], result['loc_stop_best']],
                                                      self.interpolate, 1, labxRaw=result['df']['labx'].values)
            result['caerus']['model'] = cs
        else:
            logger.warning('[method="%s"] is not supported in 1d-vector data. <return>' % (self.method))
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
        if self.method == 'peakdetect':
            # peakdetect
            dfint['labx'] = result['peakdetect']['labx_s']
            dfint['valley'] = False
            dfint['peak'] = False
            if result['peakdetect']['min_peaks_s'] is not None:
                dfint.loc[result['peakdetect']['min_peaks_s'][:, 0].astype(int), 'valley'] = True
            if result['peakdetect']['max_peaks_s'] is not None:
                dfint.loc[result['peakdetect']['max_peaks_s'][:, 0].astype(int), 'peak'] = True
        elif self.method == 'topology':
            # Topology
            dfint['labx'] = result['topology']['labx_s']
            dfint['rank'] = result['Xranked']
            dfint['score'] = result['Xdetect']
            dfint['valley'] = False
            dfint['peak'] = False

            if result['topology']['min_peaks_s'] is not None:
                dfint.loc[result['topology']['min_peaks_s'][:, 0].astype(int), 'valley'] = True
            if result['topology']['max_peaks_s'] is not None:
                dfint.loc[result['topology']['max_peaks_s'][:, 0].astype(int), 'peak'] = True

            results['persistence'] = result['persistence']
            results['Xdetect'] = result['Xdetect']
            results['Xranked'] = result['Xranked']
            results['groups0'] = result['groups0']
        elif self.method == 'caerus':
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
            if self.method == 'peakdetect':
                # peakdetect
                df['x'] = result['peakdetect']['xs']
                df['labx'] = result['peakdetect']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['peakdetect']['min_peaks'] is not None:
                    df.loc[result['peakdetect']['min_peaks'][:, 0].astype(int), 'valley'] = True
                    # df['valley'].iloc[result['peakdetect']['min_peaks'][:, 0].astype(int)] = True
                if result['peakdetect']['max_peaks'] is not None:
                    df.loc[result['peakdetect']['max_peaks'][:, 0].astype(int), 'peak'] = True
                    # df['peak'].iloc[result['peakdetect']['max_peaks'][:, 0].astype(int)] = True

            elif self.method == 'topology':
                # Topology
                df['x'] = result['topology']['xs']
                df['labx'] = result['topology']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['topology']['min_peaks'] is not None:
                    df.loc[result['topology']['min_peaks'][:, 0].astype(int), 'valley'] = True
                    # df['valley'].iloc[result['topology']['min_peaks'][:, 0].astype(int)] = True
                if result['topology']['max_peaks'] is not None:
                    # df['peak'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = True
                    df.loc[result['topology']['max_peaks'][:, 0].astype(int), 'peak'] = True

                # Store the score and ranking
                df['rank'] = 0
                df['score'] = 0.0  # Ensure float dtype

                df.loc[result['topology']['max_peaks'][:, 0].astype(int), 'rank'] = dfint.loc[
                    result['topology']['max_peaks_s'][:, 0].astype(int), 'rank'].values
                df.loc[result['topology']['max_peaks'][:, 0].astype(int), 'score'] = (
                    dfint.loc[result['topology']['max_peaks_s'][:, 0].astype(int), 'score'].values.astype(float)
                )
                # df['rank'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = dfint['rank'].iloc[
                #     result['topology']['max_peaks_s'][:, 0].astype(int)].values
                # df['score'].iloc[result['topology']['max_peaks'][:, 0].astype(int)] = dfint['score'].iloc[
                #     result['topology']['max_peaks_s'][:, 0].astype(int)].values

                # df['rank'].loc[df['peak']] = dfint['rank'].loc[dfint['peak']].values
                # df['score'].loc[df['peak']] = dfint['score'].loc[dfint['peak']].values
            if self.method == 'caerus':
                # caerus
                df['x'] = result['caerus']['xs']
                df['labx'] = result['df']['labx']
                df['valley'] = False
                df['peak'] = False
                if result['caerus']['min_peaks'] is not None:
                    df.loc[result['caerus']['min_peaks'][:, 0].astype(int), 'valley'] = True
                if result['caerus']['max_peaks'] is not None:
                    df.loc[result['caerus']['max_peaks'][:, 0].astype(int), 'peak'] = True

            # Store in results
            results['df'] = df
            results['df_interp'] = dfint
        else:
            results['df'] = dfint

        if self.method == 'caerus':
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
        """Detect peaks and valleys in 2D-array or image data.

        Detection Description
        ---------------------
        Performs peak and valley detection on 2D data (images, spatial data, or matrices).
        Applies preprocessing steps including denoising, scaling, and grayscale conversion
        before detection. Supports multiple detection algorithms optimized for 2D data.
        
        **Preprocessing Pipeline:**
        1. **Resizing** (optional): Reduce image size for faster computation
        2. **Scaling** (optional): Normalize values to [0-255] range
        3. **Grayscale conversion** (optional): Convert color images to grayscale
        4. **Denoising** (optional): Apply noise reduction filters
        
        **Detection Methods:**
        - **topology**: Persistent homology-based detection (most robust, handles noise well)
        - **mask**: Local maximum filtering for peak detection

        Parameters
        ----------
        X : array-like
            2D input data (image, spatial data, or matrix)
            
        method : str, optional (default: 'topology')
            Detection method to use:
            - 'topology': Persistent homology-based detection (recommended)
            - 'mask': Local maximum filtering

        Returns
        -------
        dict
            Dictionary containing detection results:
            
            **Common keys:**
            - 'Xraw': numpy array of original input data
            - 'Xproc': numpy array of preprocessed data
            - 'Xdetect': numpy array with detection scores (same shape as input)
            - 'Xranked': numpy array with ranked peak/valley indices (same shape as input)
            
            **For topology method:**
            - 'persistence': pandas DataFrame with persistence scores and coordinates
            - 'groups0': list of homology groups
            
            **Persistence DataFrame columns:**
            - 'x', 'y': Coordinates of detected features
            - 'birth_level': Birth level in persistence diagram
            - 'death_level': Death level in persistence diagram
            - 'score': Persistence scores (higher = more significant)
            - 'peak': Boolean indicating peak locations
            - 'valley': Boolean indicating valley locations

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # Basic 2D peak detection
        >>> X = fp.import_example('2dpeaks')
        >>> fp = findpeaks(method='topology', limit=0.1)
        >>> results = fp.peaks2d(X)
        >>> 
        >>> # With preprocessing
        >>> fp = findpeaks(method='topology', imsize=(300,300), denoise='fastnl')
        >>> results = fp.peaks2d(X)
        >>> print(f"Found {len(results['persistence'])} significant features")

        Notes
        -----
        - The topology method provides the most robust detection with mathematical stability
        - Preprocessing steps can significantly improve detection quality on noisy data
        - Persistence scores quantify the significance of detected features
        - Results are automatically stored in the object for plotting and analysis
        - Color images are automatically converted to grayscale for topology method

        See Also
        --------
        fit : Main detection method for both 1D and 2D data
        peaks1d : 1D peak detection method
        preprocessing : Detailed preprocessing pipeline
        plot : Visualize detection results
        """
        if method is None: method = 'topology'
        # Set image as float and set negative values to 0
        X = np.clip(X.astype(float), 0, None)
        
        self.method = method
        self.type = 'peaks2d'
        logger.debug('Finding peaks in 2d-array using %s method..' % (self.method))
        if (not self.togray) and (len(X.shape) == 3) and (self.method == 'topology'): logger.error(
            'Topology method requires 2d-array. Your input is 3d. Hint: set togray=True.')

        # Preprocessing the image
        Xproc = self.preprocessing(X, showfig=False)
        # Compute peaks based on method
        if method == 'topology':
            # Compute persistence based on topology method
            result = stats.topology2d(Xproc, limit=self.limit, whitelist=self.whitelist)
            # result = stats.topology(Xproc, limit=self.limit)
        elif method == 'mask':
            # Compute peaks using local maximum filter.
            result = stats.mask(Xproc, limit=self.limit)
        else:
            logger.warning(
                '[method="%s"] is not supported in 2d-array (image) data. <return>' % (
                    self.method))
            return None

        # Store
        self.results, self.args = self._store2d(X, Xproc, result)
        # Return
        logger.info('Fin.')
        return self.results

    # Store 2D-array
    def _store2d(self, X, Xproc, result):
        # Store results
        results = {}
        results['Xraw'] = X
        results['Xproc'] = Xproc

        # Store method specific results
        if self.method == 'topology':
            # results['topology'] = result
            results['Xdetect'] = result['Xdetect']
            results['Xranked'] = result['Xranked']
            results['persistence'] = result['persistence']
            results['persistence'].reset_index(inplace=True, drop=True)
            # results['peak'] = result['peak'] # These values are incorrect when using 2d
            # results['valley'] = result['valley'] # These values are incorrect when using 2d
            results['groups0'] = result['groups0']
        if self.method == 'mask':
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
        """Apply preprocessing pipeline to 2D array (image) data.

        Preprocessing Description
        -------------------------
        Performs a series of optional preprocessing steps to prepare 2D data for peak detection.
        The preprocessing pipeline can significantly improve detection quality, especially
        for noisy or complex images.

        **Preprocessing Steps (in order):**
        1. **Resizing**: Reduce image dimensions for faster computation
        2. **Scaling**: Normalize pixel values to [0-255] range for consistent processing
        3. **Grayscale conversion**: Convert color images to grayscale (required for topology method)
        4. **Denoising**: Apply noise reduction filters to improve detection quality

        Parameters
        ----------
        X : numpy.ndarray
            Input 2D data (image, spatial data, or matrix)
            
        showfig : bool, optional (default: False)
            If True, displays intermediate preprocessing steps as subplots.
            Useful for understanding how each step affects the data.

        Returns
        -------
        numpy.ndarray
            Preprocessed 2D array ready for peak detection

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # Basic preprocessing
        >>> fp = findpeaks(scale=True, togray=True, denoise='fastnl')
        >>> X_processed = fp.preprocessing(X)
        >>> 
        >>> # With visualization
        >>> fp.preprocessing(X, showfig=True)

        Notes
        -----
        - Preprocessing steps are applied in the order listed above
        - Each step is optional and controlled by initialization parameters
        - Grayscale conversion is required for topology method
        - Denoising can significantly improve detection on noisy data
        - Resizing reduces computation time but may lose fine details

        See Also
        --------
        peaks2d : 2D peak detection with automatic preprocessing
        plot_preprocessing : Visualize preprocessing steps
        """

        if showfig:
            # Number of axis to create:
            nplots = 1 + (self.imsize is not None) + self.scale + self.togray + (self.denoise is not None)
            fig, ax = plt.subplots(1, nplots, figsize=self.figsize)
            iax = 0
            # Plot RAW input image
            X=(X.astype(np.uint8).copy() if self.togray else X)
            ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
            ax[iax].grid(False)
            ax[iax].set_title('Input\nRange: [%.3g,%.3g]' % (X.min(), X.max()))
            iax = iax + 1
        # Resize
        if self.imsize:
            X = stats.resize(X, size=self.imsize)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Resize\n(%s,%s)' % (self.imsize))
                iax = iax + 1
        # Scaling color range between [0,255]
        # The or functiona is  necessary because OpenCV's fastNlMeansDenoising and bilateralFilter
        if self.scale or (self.denoise in ['fastnl', 'bilateral'] and X.dtype != np.uint8):
            X = stats.scale(X)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Scale\nRange: [%.3g %.3g]' % (X.min(), X.max()))
                iax = iax + 1
        # Convert to gray image
        if self.togray:
            X = stats.togray(X)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Color conversion\nGray')
                iax = iax + 1
        # Denoise
        if self.denoise is not None:
            X = stats.denoise(X, method=self.denoise, window=self.window, cu=self.cu)
            if showfig:
                # plt.figure(figsize=self.figsize)
                ax[iax].imshow(X, cmap=('gray_r' if self.togray else None))
                ax[iax].grid(False)
                ax[iax].set_title('Denoise\n' + self.method)
                iax = iax + 1
        # Return
        return X

    # Pre-processing
    def imread(self, path):
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
            logger.info('Downloading from github source: [%s]' % (path))
            response = requests.get(path)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            X = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        elif os.path.isfile(path):
            logger.info('Import [%s]' % (path))
            X = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # Return
        return X

    # %% Plotting
    def plot(self,
             limit=None,
             legend=True,
             figsize=None,
             cmap=None,
             text=True,
             s=None,
             marker='x',
             color='#FF0000',
             xlabel='x-axis',
             ylabel='y-axis',
             figure_order='vertical',
             fontsize=18):
        """Plot peak detection results.

        Plot Description
        ----------------
        Creates visualizations of peak detection results. Automatically determines
        the appropriate plot type based on the data dimensionality (1D or 2D).
        
        **For 1D data**: Creates line plots with peak/valley markers
        **For 2D data**: Creates image plots with peak/valley annotations

        Parameters
        ----------
        limit : float, optional (default: None)
            Persistence threshold for filtering results. Only features with
            persistence > limit are displayed (topology method only).
            
        legend : bool, optional (default: True)
            Whether to display the plot legend.
            
        figsize : tuple, optional (default: None)
            Figure size as (width, height) in inches. If None, uses default size.
            
        cmap : str or matplotlib.colors.Colormap, optional (default: None)
            Colormap for 2D plots. If None, automatically chooses based on data type.
            Common options: 'gray', 'hot', 'viridis', plt.cm.hot_r
            
        text : bool, optional (default: True)
            Whether to display peak/valley labels on 2D plots.
            
        s : float, optional (default: None)
            Marker size for peak/valley indicators. If None, automatically sizes
            based on peak significance.
            
        marker : str, optional (default: 'x')
            Marker style for peaks. Common options: 'x', 'o', '+', '*'
            
        color : str, optional (default: '#FF0000')
            Color for peak/valley markers (hex color code).
            
        xlabel : str, optional (default: 'x-axis')
            Label for the x-axis.
            
        ylabel : str, optional (default: 'y-axis')
            Label for the y-axis.
            
        figure_order : str, optional (default: 'vertical')
            Layout direction for subplots: 'vertical' or 'horizontal'.
            
        fontsize : int, optional (default: 18)
            Font size for text labels and axis labels.

        Returns
        -------
        tuple or None
            For 1D data: (ax1, ax2) - matplotlib axes objects
            For 2D data: (ax1, ax2, ax3) - matplotlib axes objects
            Returns None if no results are available to plot

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> 
        >>> # Basic plotting
        >>> fp = findpeaks(method='topology')
        >>> results = fp.fit(X)
        >>> fp.plot()
        >>> 
        >>> # Customized plotting
        >>> fp.plot(limit=0.1, marker='o', color='blue', fontsize=14)

        Notes
        -----
        - Must call fit() before plotting
        - For 1D data, shows both original and interpolated results (if applicable)
        - For 2D data, shows input, processed, and detection results
        - Peak/valley markers are automatically sized based on significance
        - Use limit parameter to filter out low-significance features

        See Also
        --------
        plot1d : 1D-specific plotting
        plot_mask : 2D-specific plotting with masking
        plot_persistence : Persistence diagram visualization
        plot_mesh : 3D mesh visualization
        """
        if not hasattr(self, 'results'):
            logger.warning('Nothing to plot. <return>')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']

        if self.args['type'] == 'peaks1d':
            fig_axis = self.plot1d(legend=legend, figsize=figsize, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)
        elif self.args['type'] == 'peaks2d':
            # fig_axis = self.plot2d(figsize=figsize)
            fig_axis = self.plot_mask(figsize=figsize, cmap=cmap, text=text, limit=limit, s=s, marker=marker, color=color, figure_order=figure_order, fontsize=fontsize)
        else:
            logger.warning('Nothing to plot for %s' % (self.args['type']))
            return None

        # Return
        return fig_axis

    def plot1d(
        self,
        legend=True,
        figsize=None,
        xlabel='x-axis',
        ylabel='y-axis',
        fontsize=18,
        params_line={'color':None, 'linewidth':2},
        params_peak_marker={'marker': 'x', 'color':'red', 's':120, 'edgecolors':'red', 'linewidths':3}, 
        params_valley_marker={'marker': 'o', 'color':'blue', 's':120, 'edgecolors':'lightblue', 'linewidths':3},
        ):
        """Plot the 1D results.

        Parameters
        ----------
        legend : bool, (default: True)
            Show the legend.
        figsize : (int, int), (default: None)
            (width, height) in inches.
        fontsize: int (default: 10)
            Font size for the text labels on the plot.
        params_line: dict (default: {'color':None, 'linewidth':2})
            Parameters for the line plot.
        params_peak_marker: dict (default: None)
            Parameters for the peak markers.
        params_valley_marker: dict (default: None)
            Parameters for the valley markers.
        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not self.args['type'] == 'peaks1d':
            logger.debug('Requires results of 1D data <return>.')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']
        ax1, ax2 = None, None
        title = self.method

        if self.method == 'caerus':
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
            ax1 = _plot_original(df['y'].values, df['x'].values, df['labx'].values, min_peaks.astype(int),
                                 max_peaks.astype(int), title=title, figsize=figsize, legend=legend, xlabel=xlabel,
                                 ylabel=ylabel, fontsize=fontsize, params_line=params_line, params_peak_marker=params_peak_marker, params_valley_marker=params_valley_marker)

            # Make interpolated plot
            if self.interpolate is not None:
                min_peaks, max_peaks = np.array([]), np.array([])
                df_interp = self.results['df_interp']
                if np.any('valley' in self.whitelist):
                    min_peaks = df_interp['x'].loc[df_interp['valley']].values
                if np.any('peak' in self.whitelist):
                    max_peaks = df_interp['x'].loc[df_interp['peak']].values
                ax2 = _plot_original(df_interp['y'].values, df_interp['x'].values, df_interp['labx'].values,
                                     min_peaks.astype(int), max_peaks.astype(int), title=title + ' (interpolated)',
                                     figsize=figsize, legend=legend, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, params_peak_marker=params_peak_marker, params_valley_marker=params_valley_marker)
            # Return axis
            return (ax2, ax1)

    def plot2d(self, figsize=None, limit=None, figure_order='vertical', fontsize=18):
        """Plot the 2d results.

        Parameters
        ----------
        figsize : (int, int), (default: None)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not self.args['type'] == 'peaks2d':
            logger.debug('Requires results of 2D data <return>.')
            return None
        ax_method, ax_mesh = None, None
        figsize = figsize if figsize is not None else self.args['figsize']
        # Plot preprocessing steps
        self.plot_preprocessing()

        # Setup figure
        if self.method == 'mask':
            ax_method = self.plot_mask(figsize=figsize, limit=limit, figure_order=figure_order, fontsize=fontsize)
        if self.method == 'topology':
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
        if (not hasattr(self, 'results')) or (self.type == 'peaks1d'):
            logger.warning(
                'Nothing to plot. Hint: run fit(X), where X is the (image) data. <return>')
            return None

        _ = self.preprocessing(X=self.results['Xraw'], showfig=True)

    def plot_mask(self, limit=None, figsize=None, cmap=None, text=True, s=None, marker='x', color='#FF0000',
                  figure_order='vertical', alpha=0.7, fontsize=18):
        """Plot the masking.

        Parameters
        ----------
        limit : float, (default : None)
            Values > limit are set as regions of interest (ROI).
        figsize : (int, int), (default: None)
            (width, height) in inches.
        cmap : object (default : None)
            Colormap. The default is derived wether image is convert to grey or not. Other options are: plt.cm.hot_r.
        text : Bool (default : True)
            Include text to the 2D-image that shows the peaks (p-number) and valleys (v-number)
        s : size (default: None)
            Size of the marker.
        alpha: Transparancy for the overlay detected points in ax1
            0.7
        marker: str (default: 'x')
            Marker type.
        color: str (default: '#FF0000')
            Hex color of the marker.
        figure_order: str (default: 'vertical')
            Order of the subplots ('vertical' or 'horizontal').
        fontsize: int (default: 10)
            Font size for the text labels on the plot.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if (self.type == 'peaks1d'):
            logger.warning(
                'Nothing to plot. Hint: run fit(X), where X is the 2d-array (image). <return>')
            return None

        if limit is None: limit = self.limit
        # Only show above the limit
        Xdetect = self.results['Xdetect'].copy()
        if limit is not None:
            Xdetect[np.abs(Xdetect) < limit] = 0
        if cmap is None:
            cmap = 'gray' if self.args['togray'] else None
            cmap = cmap + '_r'
        # Get the index for the detected peaks/valleys
        idx_peaks = np.where(Xdetect > 0)
        idx_valleys = np.where(Xdetect < 0)

        # Setup figure
        figsize = figsize if figsize is not None else self.args['figsize']
        if figure_order == 'vertical':
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        # Plot input image
        if self.results['Xraw'].flatten().min()<0:
            # If any negative values are found (which if weird though, clip it to 0)
            Xraw = np.clip(self.results['Xraw'], 0, None).copy()
        else:
            Xraw = self.results['Xraw'].astype(np.uint8).copy()
            # imsize
            # TODO: RESIZE IMAGE
        
        Xraw = stats.resize(Xraw, size=self.imsize)
        ax1.imshow(Xraw, cmap, interpolation="nearest")
        # Get coordinates of detections
        y, x = np.where(np.abs(Xdetect) > 0  )
        # Scatter with larger markers (s = size)
        ax1.scatter(x, y, s=s, c='red', marker=marker, alpha=alpha) 
        ax1.set_title('Input')
        ax1.grid(False)


        # For vizualisation purposes, plot all absolute numbers
        Xproc = self.results['Xproc'].copy()
        Xproc[idx_peaks] = 0
        Xproc[idx_valleys] = 1
        ax2.imshow(Xproc, cmap, interpolation="nearest")
        ax2.set_title('Processed image')
        ax2.grid(False)

        # Masking
        ax3.imshow(np.abs(Xdetect), 'gray_r', interpolation="nearest")
        ax3.set_title(self.method + ' (' + str(len(np.where(Xdetect > 0)[0])) + ' peaks and ' + str(
            len(np.where(Xdetect < 0)[0])) + ' valleys)')
        ax3.grid(False)

        # Plot markers for detected peaks and valleys
        if self.results.get('persistence', None) is not None:
            # Use persistence data for topology method
            if limit is not None:
                X = self.results['persistence'].loc[self.results['persistence']['score'] > limit, :].copy()
            elif s is not None:
                X = self.results['persistence'].copy()
            else:
                # Make empty dataframe
                X = pd.DataFrame()
                if marker is not None and s is None:
                    logger.warning('Custom marker is not shown when s=None. Set s to show the marker.')

            for i in range(X.shape[0]):
                if s is None:
                    X['score'] = stats.normalize(X['score'].values, minscale=2, maxscale=10, scaler='minmax')
                else:
                    X['score'] = s

                # Plot and include marker size
                # ax1.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)
                ax2.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)
                ax3.plot(X['x'].iloc[i], X['y'].iloc[i], markersize=X['score'].iloc[i], color=color, marker=marker)
        else:
            # Plot markers for all detected peaks and valleys
            marker_size = s if s is not None else 5
            # Plot peaks
            for idx in zip(idx_peaks[0], idx_peaks[1]):
                # ax1.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)
                ax2.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)
                ax3.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)
            # Plot valleys
            for idx in zip(idx_valleys[0], idx_valleys[1]):
                # ax1.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)
                ax2.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)
                ax3.plot(idx[1], idx[0], markersize=marker_size, color=color, marker=marker)

        if text:
            texts_ax3 = []
            # Plot peaks
            for idx in tqdm(zip(idx_peaks[0], idx_peaks[1]), disable=disable_tqdm(), desc=logger.info("Annotating peaks")):
                ax2.text(idx[1], idx[0], 'p' + self.results['Xranked'][idx].astype(str), fontsize=fontsize)
                texts_ax3.append(ax3.text(idx[1], idx[0], 'p' + self.results['Xranked'][idx].astype(str), fontsize=fontsize))
            # Plot valleys
            for idx in tqdm(zip(idx_valleys[0], idx_valleys[1]), disable=disable_tqdm(), desc=logger.info("Annotating valleys")):
                ax2.text(idx[1], idx[0], 'v' + self.results['Xranked'][idx].astype(str), fontsize=fontsize)
                texts_ax3.append(ax3.text(idx[1], idx[0], 'v' + self.results['Xranked'][idx].astype(str), fontsize=fontsize))
            # Adjust text labels on ax3 to prevent overlap
            if len(texts_ax3)>0: _, _ = adjust_text(texts_ax3)

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
        """Plot the 3D-mesh.

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
        
        Example
        -------
        >>> # Import library
        >>> from findpeaks import findpeaks
        >>> #
        >>> # Initialize
        >>> fp = findpeaks(method='topology',imsize=False,scale=False,togray=False,denoise=None,params={'window': 15})
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
            logger.warning('Nothing to plot. Hint: run the fit() function. <return>')
            return None
        if self.results.get('Xproc', None) is None:
            logger.warning(
                'These analysis do not support mesh plotting. This may be caused because your are analysing 1D.')
            return None

        figsize = figsize if figsize is not None else self.args['figsize']
        logger.debug('Plotting 3d-mesh..')
        ax1, ax2 = None, None
        if savepath is not None:
            savepath = str.replace(savepath, ',', '_')
            savepath = str.replace(savepath, '=', '_')

        # Compute meshgrid
        Z = self.results['Xproc'].copy()
        X, Y = np.mgrid[0:Z.shape[0], 0:Z.shape[1]]
        # To limit on the X and Y axis, we need to create a trick by setting all values to nan in the Z-axis that should be limited.
        if xlim is not None:
            if xlim[0] is not None: Z[X < xlim[0]] = np.nan
            if xlim[1] is not None: Z[X > xlim[1]] = np.nan
        if ylim is not None:
            if ylim[0] is not None: Z[Y < ylim[0]] = np.nan
            if ylim[1] is not None: Z[Y > ylim[1]] = np.nan
        if zlim is not None:
            if zlim[0] is not None: Z[Z < zlim[0]] = np.nan
            if zlim[1] is not None: Z[Z > zlim[1]] = np.nan

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
                logger.info('Saving wireframe to disk..')
                fig.savefig(savepath)

        if surface:
            # Plot the figure
            fig = plt.figure(figsize=figsize)
            ax2 = fig.add_subplot(projection='3d')
            ax2 = fig.gca()
            ax2.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, cmap=cmap, linewidth=0, shade=True,
                             antialiased=False)
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
                logger.info('Saving surface to disk..')
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

    def plot_persistence(self, figsize=(20, 8), fontsize_ax1=14, fontsize_ax2=14, xlabel='x-axis', ylabel='y-axis', s=20, marker='x', color='#FF0000'):
        """Plot the homology-peristence.

        Parameters
        ----------
        figsize : (int, int), (default: None)
            (width, height) in inches.
        fontsize_ax1 : int, (default: 14)
            Font size for the labels in the left figure. Choose None for no text-labels.
        fontsize_ax2 : int, (default: 14)
            Font size for the labels in the right figure. Choose None for no text-labels.

        Returns
        -------
        ax1 : object
            Figure axis 1.
        ax2 : object
            Figure axis 2.

        """
        if (self.method != 'topology') or (not hasattr(self, 'results')) or len(
                self.results['persistence']['birth_level'].values) <= 0:
            logger.warning(
                'Nothing to plot. Hint: run the .fit(method="topology") function. <return>')
            return None

        # Setup figure
        figsize = figsize if figsize is not None else self.args['figsize']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Create the persistence ax2
        ax2 = self._plot_persistence_ax2(fontsize_ax2, ax2, s=s, marker=marker, color=color)
        # Create the persistence ax1
        ax1, ax2 = self._plot_persistence_ax1(fontsize_ax1, ax1, ax2, figsize, xlabel, ylabel, s=s, marker=marker, color=color)
        # Plot
        plt.show()
        # Return
        return ax1, ax2

    def _plot_persistence_ax1(self, fontsize, ax1, ax2, figsize, xlabel, ylabel, s=20, marker='x', color='#FF0000'):
        if self.args['type'] == 'peaks1d':
            # Attach the ranking-labels
            if fontsize is not None:
                x = self.results['df']['x'].values
                y = self.results['df']['y'].values
                idx = np.where(self.results['df']['rank'] > 0)[0]
                for i in tqdm(idx, disable=disable_tqdm(), desc=logger.info("Plotting persistence axis 1")):
                    ax1.text(x[i], (y[i] + y[i] * 0.01), str(self.results['df']['rank'].iloc[i]), color='b', fontsize=fontsize)

            # minpers = 0
            min_peaks, max_peaks = np.array([]), np.array([])
            if np.any('valley' in self.whitelist):
                min_peaks = self.results['df']['x'].loc[self.results['df']['valley']].values
            if np.any('peak' in self.whitelist):
                max_peaks = self.results['df']['x'].loc[self.results['df']['peak']].values

            # Make the plot
            ax1 = _plot_original(self.results['df']['y'].values, self.results['df']['x'].values,
                                 self.results['df']['labx'].values, min_peaks.astype(int), max_peaks.astype(int),
                                 title='Persistence', figsize=figsize, legend=True, ax=ax1, xlabel=xlabel,
                                 ylabel=ylabel)
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
            logger.info('Plotting loci of birth..')
            ax1.set_title("Loci of births")
            texts = []
            for i, homclass in tqdm(enumerate(self.results['groups0']), disable=disable_tqdm(), desc=logger.info("Plotting loci of births")):
                p_birth, bl, pers, p_death = homclass
                if (self.limit is None):
                    y, x = p_birth
                    Xdetect[y, x] = i + 1
                    ax1.scatter(x, y, s=s, marker=marker, c=color)
                    ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize)
                    # texts.append(ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize))
                elif pers > self.limit:
                    y, x = p_birth
                    Xdetect[y, x] = i + 1
                    # ax1.plot([x], [y], '.', c='b')
                    ax1.scatter(x, y, s=s, marker=marker, c=color)
                    ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize)
                    # texts.append(ax1.text(x, y + 0.25, str(i), color='b', fontsize=fontsize))

            # Plot the adjusted text labels to prevent overlap. Do not adjust text in 3d plots as it will mess up the locations.
            ax1.set_xlim((0, self.results['Xproc'].shape[1]))
            ax1.set_ylim((0, self.results['Xproc'].shape[0]))
            ax1.invert_yaxis()
            plt.gca().invert_yaxis()
            ax1.grid(True)
            # Set the axis to 255-255 in ax2 because it is an image.
            ax2.plot([0, 255], [0, 255], '-', c='grey')
        return ax1, ax2

    def _plot_persistence_ax2(self, fontsize, ax2, s=20, marker='x', color='#FF0000'):
        x = self.results['persistence']['birth_level'].values
        y = self.results['persistence']['death_level'].values
        if len(x) <= 0:
            logger.debug('Nothing to plot.')
            return None

        ax2.scatter(x, y, s=s, marker=marker, c=color)
        if fontsize is not None:
            texts = []
            for i in tqdm(range(0, len(x)), disable=disable_tqdm(), desc=logger.info("Plotting persistence axis 2")):
                ax2.text(x[i], (y[i] + y[i] * 0.01), str(i + 1), color='b', fontsize=fontsize)
                texts.append(ax2.text(x[i], (y[i] + y[i] * 0.01), str(i + 1), color='b', fontsize=fontsize))
            if len(texts)>0: _, _ = adjust_text(texts)


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

        Dataset Import Descriptions
        ---------------------------
        Import one of the few datasets from github source or specify your own download url link.

        Parameters
        ----------
        data : str
            Name of datasets: "1dpeaks", "2dpeaks", "2dpeaks_image", '2dpeaks_image_2', 'btc', 'facebook'
        url : str
            url link to to dataset.
        datadir : path-like
            Directory to store downloaded datasets in. Defaults to data sub-directory
            of findpeaks install location.

        Returns
        -------
        pd.DataFrame()
            Dataset containing mixed features.

        """
        X = import_example(data=data, url=url, sep=sep, datadir=datadir)
        return X


    def check_logger(self, verbose: [str, int] = None):
        """Check the logger."""
        if verbose is not None: set_logger(verbose)
        logger.debug('DEBUG')
        logger.info('INFO')
        logger.warning('WARNING')
        logger.critical('CRITICAL')
        
# %%
def _plot_original(X,
                   xs, 
                   labx, 
                   min_peaks, 
                   max_peaks, 
                   title=None, 
                   legend=True, 
                   ax=None, 
                   figsize=(15, 8), 
                   xlabel=None,
                   ylabel=None, 
                   fontsize=18,
                   params_line={'color':None, 'linewidth':2},
                   params_peak_marker={'marker': 'x', 'color':'red', 's':120, 'edgecolors':'lightred', 'linewidths':2},
                   params_valley_marker={'marker': 'o', 'color':'blue', 's':120, 'edgecolors':'lightblue', 'linewidths':2},
                   ):

    uilabx = np.unique(labx)
    uilabx = uilabx[~np.isnan(uilabx)]

    # Set the default for the peaks and valleys
    params_peak_marker_default = dict(color='red', s=120, edgecolors='red', linewidths=3)
    params_peak_marker = {**params_peak_marker_default, **params_peak_marker}
    params_valley_marker_default = dict(color='blue', s=100, edgecolors='blue', linewidths=2)
    params_valley_marker = {**params_valley_marker_default, **params_valley_marker} 
    params_line_default = dict(color=None, linewidth=2)
    params_line = {**params_line_default, **params_line} 

    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    params_line_black = params_line.copy()
    params_line_black['color'] = 'black'
    ax.plot(xs, X, **params_line_black)
    
    if params_peak_marker.get('marker', 'x') in ['x', '+', '.', '|', '_']:
        params_peak_marker.pop('edgecolors', None)
    ax.scatter(max_peaks, X[max_peaks], label='Peak', **params_peak_marker)

    if params_valley_marker.get('marker', 'o') in ['x', '+', '.', '|', '_']:
        params_valley_marker.pop('edgecolors', None)
    ax.scatter(min_peaks, X[min_peaks], label='Valley', **params_valley_marker)

    # Color each detected label
    s = np.arange(0, len(X))
    for i in uilabx:
        idx = (labx == i)
        ax.plot(s[idx], X[idx], **params_line)
        # plt.plot(s[idx], X[idx], label='peak' + str(i))

    if legend: ax.legend(loc=0)
    ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.show()
    return ax


# %% Import example dataset from github.
def import_example(data='2dpeaks', url=None, sep=';', datadir=None):
    """Import example dataset from github source.

    Dataset Import Description
    --------------------------
    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: "2dpeaks" or "2dpeaks_image" or '2dpeaks_image_2'
    url : str
        url link to to dataset.
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
            logger.warning('Could not determine filename to download <return>.')
            return None
        data, _ = os.path.splitext(fn)
    elif data == '2dpeaks_image' or data == '2dpeaks_image_2':
        url = 'https://erdogant.github.io/datasets/' + data + '.png'
        fn = data + '.png'
    elif data == '2dpeaks':
        url = 'https://erdogant.github.io/datasets/' + data + '.zip'
        fn = "2dpeaks.zip"
    elif data == '1dpeaks':
        # x = [0,   13,  22,  30,  35,  38,   42,   51,   57,   67,  73,   75,  89,   126,  141,  150,  200 ]
        y = [1.5, 0.8, 1.2, 0.2, 0.4, 0.39, 0.42, 0.22, 0.23, 0.1, 0.11, 0.1, 0.14, 0.09, 0.04, 0.02, 0.01]
        # X = np.c_[x, y]
        return y
    elif (data == 'btc') or (data == 'facebook'):
        from caerus import caerus
        cs = caerus()
        X = cs.download_example(name=data, verbose=0)
        return X
    else:
        logger.warning('WARNING: Nothing to download. <return>')
        return None

    if datadir is None: datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(datadir, fn)
    if not os.path.isdir(datadir):
        os.makedirs(datadir, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading from github source: [%s]' % (url))
        r = requests.get(url, stream=True)
        with open(PATH_TO_DATA, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                fd.write(chunk)

    # Import local dataset
    logger.info('Import [%s]' % (PATH_TO_DATA))
    if data == '2dpeaks_image' or data == '2dpeaks_image_2':
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


# %% Verbosity
# =============================================================================
# Functions for verbosity
# =============================================================================
def convert_verbose_to_old(verbose):
    """Convert new verbosity to the old ones."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if isinstance(verbose, str) or verbose>=10:
        status_map = {
            60: 0, 'silent': 0, 'off': 0, 'no': 0, None: 0,
            40: 1, 'error': 1, 'critical': 1,
            30: 2, 'warning': 2,
            20: 3, 'info': 3,
            10: 4, 'debug': 4}
        return status_map.get(verbose, 0)
    else:
        return verbose

def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[findpeaks] WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose

def get_logger():
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info', return_status: bool = False):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : str or int, optional, default='info' (20)
        Logging verbosity level. Possible values:
        - 0, 60, None, 'silent', 'off', 'no' : no messages.
        - 10, 'debug' : debug level and above.
        - 20, 'info' : info level and above.
        - 30, 'warning' : warning level and above.
        - 50, 'critical' : critical level and above.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {
            'silent': logging.CRITICAL + 10,
            'off': logging.CRITICAL + 10,
            'no': logging.CRITICAL + 10,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        verbose = levels[verbose]

    # Set up basic configuration if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=verbose, format='[{asctime}] [{name}] [{levelname}] {msg}', style='{', datefmt='%d-%m-%Y %H:%M:%S')
    else:
        # Set the root logger level to control all child loggers
        logging.getLogger().setLevel(verbose)
    
    # Also set the specific logger level
    logger.setLevel(verbose)

    if return_status:
        return verbose

def check_logger(verbose: [str, int] = None):
    """Check the logger."""
    if verbose is not None: set_logger(verbose)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.critical('CRITICAL')