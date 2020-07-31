# ----------------------------------------------------
# Name        : findpeaks.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# github      : https://github.com/erdogant/findpeaks
# Licence     : See Licences
# ----------------------------------------------------

import findpeaks.utils.stats as stats
from findpeaks.utils.smoothline import interpolate_line1d
from peakdetect import peakdetect
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import wget
import os

class findpeaks():
    def __init__(self, lookahead=200, interpolate=None, mask=0, imsize=None, scale=True, togray=True, denoise='fastnl', window=3, cu=0.25, figsize=(15, 8), verbose=3):
        """Initialize findpeaks parameters.

        Parameters 1D
        -------------
        X : array-like RGB or 1D-array
            Input image data.
        lookahead : int, (default : 200)
            Looking ahead for peaks. For very small 1d arrays (such as up to 50 datapoints), use low numbers: 1 or 2.
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
        denoise : string, (default : 'fastnl', None to disable)
            Filtering method to remove noise: [None, 'fastnl','bilateral','lee','lee_enhanced','kuan','frost','median','mean']
        window : int, (default : 3)
            Denoising window. Increasing the window size may removes noise better but may also removes details of image in certain denoising methods.
        cu : float, (default: 0.25)
            The noise variation coefficient, applies for methods: ['kuan','lee','lee_enhanced']
        togray : bool, (default : False)
            Conversion to gray scale.
        imsize : tuple, (default : None)
            size to desired (width,length).
        Verbose : int (default : 3)
            Print to screen. 0: None, 1: Error, 2: Warning, 3: Info, 4: Debug, 5: Trace.

        """
        # Store in object
        if lookahead==None: lookahead=1
        lookahead = np.maximum(1, lookahead)

        self.lookahead = lookahead
        self.interpolate = interpolate
        self.mask = mask
        self.imsize = imsize
        self.scale = scale
        self.togray = togray
        self.denoise = denoise
        self.window = window
        self.cu = cu
        self.figsize = figsize
        self.verbose = verbose

    def fit(self, X, x=None):
        """Detect peaks and valleys in a 1D vector or 2D-array (image).

        Parameters
        ----------
        X : array-like 1D vector.
            Input data.
        x : array-like 1D vector.
            Coordinates of the x-axis.

        Returns
        -------
        dict. Output depends wether peaks1d or peaks2d is used.
        1dpeaks:
            df : DataFrame
                Results based on the input data.
                    x: x-coordinates
                    y: y-coordinates
                    labx: assigned label
                    valley: detected valley
                    peak: detected peak
                    labx_topology: assigned label based on topology method
                    valley_topology: detected valley based on topology method
                    peak_topology: detected peak based on topology method
            df_interp : DataFrame
                Results based on the interpolated data.
                    x: x-coordinates
                    y: y-coordinates
                    labx: assigned label
                    valley: detected valley
                    peak: detected peak
                    labx_topology: assigned label based on topology method
                    valley_topology: detected valley based on topology method
                    peak_topology: detected peak based on topology method
        2dpeaks:
            dict:
                Xraw: Input image
                Xproc: Processed image
                Xmask: detected peaks using masking method
                persitance: detected peaks using topology method

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
        >>> fp = findpeaks(denoise=30, imsize=(300,300))
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
        """Detection of peaks in 1D array.

        Parameters
        ----------
        X : array-like 1D vector.
            Input data.
        x : array-like 1D vector.
            Coordinates of the x-axis.

        Returns
        -------
        dict with DataFrames.
            df : DataFrame
                Results based on the input data.
                    x: x-coordinates
                    y: y-coordinates
                    labx: assigned label
                    valley: detected valley
                    peak: detected peak
                    labx_topology: assigned label based on topology method
                    valley_topology: detected valley based on topology method
                    peak_topology: detected peak based on topology method
            df_interp : DataFrame
                Results based on the interpolated data.
                    x: x-coordinates
                    y: y-coordinates
                    labx: assigned label
                    valley: detected valley
                    peak: detected peak
                    labx_topology: assigned label based on topology method
                    valley_topology: detected valley based on topology method
                    peak_topology: detected peak based on topology method

        Examples
        --------
        >>> from findpeaks import findpeaks
        >>> X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
        >>> fp = findpeaks(interpolate=10, lookahead=1)
        >>> results = fp.fit(X)
        >>> fp.plot()

        """
        X = np.array(X)
        Xraw = X.copy()
        # Interpolation
        if self.interpolate: X = interpolate_line1d(X, nboost=len(X) * self.interpolate, method=2, showfig=False)

        # Peak detect
        max_peaks, min_peaks = peakdetect(np.array(X), lookahead=self.lookahead)
        # Post processing for the peak-detect
        results_peaksdetect = stats._post_processing(X, Xraw, min_peaks, max_peaks, self.interpolate, self.lookahead)

        # Compute persistance using toplogy method
        persist_score, max_peaks_p, min_peaks_p, persistence_scores = stats.persistence(np.c_[X, X])
        # Post processing for the topology method
        results_topology = stats._post_processing(X, Xraw, min_peaks_p, max_peaks_p, self.interpolate, 1, persistence_scores=persistence_scores)

        # Store
        self.results, self.args = self._store1d(X, Xraw, x, persist_score, results_peaksdetect, results_topology)
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
        >>> fp = findpeaks(denoise='fastnl', window=30, imsize=(300,300))
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
        # Compute persistance based on topology method
        pers_score = stats._topology(Xproc)
        # Compute peaks using local maximum filter.
        Xmask = stats._mask(Xproc, mask=self.mask)
        # Store
        self.results, self.args = self._store2d(X, Xproc, Xmask, pers_score)
        # Return
        if self.verbose>=3: print('[findpeaks] >Fin.')
        return self.results

    # Store 1D vector
    def _store1d(self, X, Xraw, xs, persist_score, res_peakd, results_topology):
        if xs is None: xs = np.arange(0, len(X))
        results = {}
        # Interpolated data
        dfint = pd.DataFrame()
        dfint['x'] = xs
        dfint['y'] = X
        # peakdetect
        dfint['labx'] = res_peakd['labx_s']
        dfint['valley'] = False
        dfint['valley'].iloc[res_peakd['min_peaks_s'][:, 0].astype(int)] = True
        dfint['peak'] = False
        dfint['peak'].iloc[res_peakd['max_peaks_s'][:, 0].astype(int)] = True
        # Topology
        dfint['labx_topology'] = results_topology['labx_s']
        dfint['valley_topology'] = False
        dfint['valley_topology'].iloc[results_topology['min_peaks_s'][:, 0].astype(int)] = True
        dfint['peak_topology'] = False
        dfint['peak_topology'].iloc[results_topology['max_peaks_s'][:, 0].astype(int)] = True
        dfint['persistence'] = np.nan
        # dfint['persistence'].iloc[results_topology['max_peaks_s'][:, 0].astype(int)] = peak_pers_scores

        if self.interpolate:
            # As for the input data
            df = pd.DataFrame()
            df['x'] = res_peakd['xs']
            df['y'] = Xraw
            # peakdetect
            df['labx'] = res_peakd['labx']
            df['valley'] = False
            df['valley'].iloc[res_peakd['min_peaks'][:, 0].astype(int)] = True
            df['peak'] = False
            df['peak'].iloc[res_peakd['max_peaks'][:, 0].astype(int)] = True
            # Topology
            df['labx_topology'] = results_topology['labx']
            df['valley_topology'] = False
            df['valley_topology'].iloc[results_topology['min_peaks'][:, 0].astype(int)] = True
            df['peak_topology'] = False
            df['peak_topology'].iloc[results_topology['max_peaks'][:, 0].astype(int)] = True
            # Store in results
            results['df'] = df
            results['df_interp'] = dfint
        else:
            results['df'] = dfint

        results['topology'] = persist_score

        # Arguments
        args = {}
        args['lookahead'] = self.lookahead
        args['interpolate'] = self.interpolate
        args['figsize'] = self.figsize
        args['type'] = 'peaks1d'
        # Return
        return results, args

    # Store 2D-array
    def _store2d(self, X, Xproc, Xmask, pers_score):
        results = {}
        results['Xraw'] = X
        results['Xproc'] = Xproc
        results['Xmask'] = Xmask
        results['topology'] = pers_score
        # Store arguments
        args = {}
        args['mask'] = self.mask
        args['scale'] = self.scale
        args['denoise'] = self.denoise
        args['togray'] = self.togray
        args['imsize'] = self.imsize
        args['figsize'] = self.figsize
        args['type'] = 'peaks2d'
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
        if self.imsize:
            X = stats.resize(X, size=self.imsize)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
                plt.grid(False)
        # Scaling
        if self.scale:
            X = stats.scale(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X)
                plt.grid(False)
        # Convert to gray image
        if self.togray:
            X = stats.togray(X, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X, cmap=('gray' if self.togray else None))
                plt.grid(False)
        # Denoising
        if self.denoise is not None:
            X = stats.denoise(X, method=self.denoise, window=self.window, cu=self.cu, verbose=self.verbose)
            if showfig:
                plt.figure(figsize=self.figsize)
                plt.imshow(X, cmap=('gray' if self.togray else None))
                plt.grid(False)
        # Return
        return X

    # %% Plotting
    def plot(self, method=None, legend=True, figsize=None):
        """Plot results.

        Parameters
        ----------
        method : String, default : None or 'peakdetect'
            plot the results for method: 'topology', 'peakdetect'
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

        if self.args['type']=='peaks1d':
            fig_axis = self.plot1d(method=method, legend=legend, figsize=figsize)
        elif self.args['type']=='peaks2d':
            fig_axis = self.plot2d(figsize=figsize)
        else:
            print('[findpeaks] Nothing to plot for %s' %(self.args['type']))
            return None

        # Return
        return fig_axis

    def plot1d(self, method=None, legend=True, figsize=None):
        """Plot the 1D results.

        Parameters
        ----------
        method : String, default : None or 'peakdetect'
            plot the results for method: 'topology', 'peakdetect'
        figsize : (int, int), optional, default: (15, 8)
            (width, height) in inches.

        Returns
        -------
        fig_axis : tuple containing axis for each figure.

        """
        if not self.args['type']=='peaks1d':
            print('[findpeaks] >Requires results of 1D data <return>.')
            return None
        
        figsize = figsize if figsize is not None else self.args['figsize']
        ax1, ax2 = None, None

        # Select the data to plot
        if (method is None) or (method=='peakdetect'):
            title='peakdetect'
            df = self.results['df'][['x', 'y', 'labx', 'valley', 'peak']]
            if self.interpolate is not None:
                df_interp = self.results['df_interp'][['x', 'y', 'labx', 'valley', 'peak']]
        else:
            title=method
            df = self.results['df'][['x','y','labx_topology','valley_topology','peak_topology']]
            df.rename(columns={'labx_topology': 'labx', 'valley_topology': 'valley', 'peak_topology': 'peak'}, inplace=True)
            if self.interpolate is not None:
                df_interp = self.results['df_interp'][['x','y','labx_topology','valley_topology','peak_topology']]
                df_interp.rename(columns={'labx_topology': 'labx', 'valley_topology': 'valley', 'peak_topology': 'peak'}, inplace=True)

        # Make plot
        min_peaks = df['x'].loc[df['valley']].values
        max_peaks = df['x'].loc[df['peak']].values
        ax1 = _plot_original(df['y'].values, df['x'].values, df['labx'].values, min_peaks.astype(int), max_peaks.astype(int), title=title, figsize=figsize, legend=legend)

        # Make interpolated plot
        if self.interpolate is not None:
            min_peaks = df_interp['x'].loc[df_interp['valley']].values
            max_peaks = df_interp['x'].loc[df_interp['peak']].values
            ax2 = _plot_original(df_interp['y'].values, df_interp['x'].values, df_interp['labx'].values, min_peaks.astype(int), max_peaks.astype(int), title=title + ' (interpolated)', figsize=figsize, legend=legend)
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
        if not self.args['type']=='peaks2d':
            print('[findpeaks] >Requires results of 2D data <return>.')
            return None
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
        ax1.set_title('Original')
        ax1.grid(False)

        # Preprocessing
        ax2.imshow(self.results['Xproc'], cmap=cmap, interpolation="nearest")
        ax2.set_title('Processed image')
        ax2.grid(False)

        # Masking
        ax3.imshow(self.results['Xmask'], cmap=cmap, interpolation="nearest")
        ax3.set_title('After Masking')
        ax3.grid(False)

        # Return
        return (ax1, ax2, ax3)

    def plot_mesh(self, wireframe=True, surface=True, rstride=2, cstride=2, cmap=plt.cm.hot_r, title='', figsize=None, view=None, savepath=None):
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
            Rotate the mesh plot.
            (0, 0) : y vs z
            (0, 90) : x vs z
            (90, 0) : y vs x
            (90, 90) : x vs y
        cmap : object
            Colormap. The default is plt.cm.hot_r.
        savepath : bool (default : None)
            Path with filename to save the figure, eg: './tmp/my_image.png'
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
        ax1, ax2 = None, None
        if savepath is not None:
            savepath = str.replace(savepath, ',', '_')
            # savepath = str.replace(savepath, ':', '_')
            savepath = str.replace(savepath, '=', '_')
            # savepath = str.replace(savepath, ' ', '_')

        # Compute meshgrid
        xx, yy = np.mgrid[0:self.results['Xproc'].shape[0], 0:self.results['Xproc'].shape[1]]

        # Plot the figure
        if wireframe:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.gca(projection='3d')
            ax1.plot_wireframe(xx, yy, self.results['Xproc'], rstride=rstride, cstride=cstride, linewidth=0.8)
            ax1.set_xlabel('x-axis')
            ax1.set_ylabel('y-axis')
            ax1.set_zlabel('z-axis')
            if view is not None:
                ax1.view_init(view[0], view[1])
                # ax1.view_init(50, -10) # x vs y
            ax1.set_title(title)
            plt.show()
            if savepath is not None:
                if self.verbose>=3: print('[findpeaks] >Saving wireframe to disk..')
                fig.savefig(savepath)

        if surface:
            # Plot the figure
            fig = plt.figure(figsize=figsize)
            ax2 = fig.gca(projection='3d')
            ax2.plot_surface(xx, yy, self.results['Xproc'], rstride=rstride, cstride=cstride, cmap=cmap, linewidth=0, shade=True, antialiased=False)
            if view is not None:
                ax2.view_init(view[0], view[1])
            ax2.set_xlabel('x-axis')
            ax2.set_ylabel('y-axis')
            ax2.set_zlabel('z-axis')
            ax2.set_title(title)
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

    def plot_peristence(self, figsize=None, verbose=3):
        figsize = figsize if figsize is not None else self.args['figsize']
        if not hasattr(self, 'results'):
            if verbose>=3: print('[findpeaks] >Nothing to plot. Hint: run the fit() function.')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if self.args['type']=='peaks1d':
            minpers = 0
            # X = np.array(self.results['topology'])[:,2]
            min_peaks = self.results['df']['x'].loc[self.results['df']['valley_topology']].values
            max_peaks = self.results['df']['x'].loc[self.results['df']['peak_topology']].values
            ax1 = _plot_original(self.results['df']['y'].values, self.results['df']['x'].values, self.results['df']['labx_topology'].values, min_peaks.astype(int), max_peaks.astype(int), title='Persistance', figsize=figsize, legend=True, ax=ax1)
        else:
            # X = self.results['Xproc']
            # Make the figure
            minscore = 20
            minpers = 1
            # Plot the detected loci
            if verbose>=3: print('[findpeaks] >Plotting loci of birth..')
            ax1.set_title("Loci of births")
            for i, homclass in tqdm(enumerate(self.results['topology'])):
                p_birth, bl, pers, p_death = homclass
                if pers > minscore:
                    y, x = p_birth
                    ax1.plot([x], [y], '.', c='b')
                    ax1.text(x, y + 0.25, str(i + 1), color='b')

            ax1.set_xlim((0, self.results['Xproc'].shape[1]))
            ax1.set_ylim((0, self.results['Xproc'].shape[0]))
            ax1.invert_yaxis()
            plt.gca().invert_yaxis()
            ax1.grid(True)
            ax2.plot([0, 255], [0, 255], '-', c='grey')

        # Plot the persistence
        if verbose>=3: print('[findpeaks] >Plotting Peristence..')
        ax2.set_title("Peristence diagram")
        xcoord = []
        ycoord = []
        perssc = []
        for i, homclass in tqdm(enumerate(self.results['topology'])):
            p_birth, bl, pers, p_death = homclass
            if pers > minpers:
                x, y = bl, (bl - pers)
                xcoord.append(x)
                ycoord.append(y)
                perssc.append(pers)
                ax2.plot([x], [y], '.', c='b')
                ax2.text(x, (y + y * 0.01), str(i + 1), color='b')

        X = xcoord + ycoord
        ax2.plot([np.min(X), np.max(X)], [np.min(X), np.max(X)], '-', c='grey')
        ax2.set_xlabel("Birth level")
        ax2.set_ylabel("Death level")
        ax2.set_xlim((np.min(X), np.max(X)))
        ax2.set_ylim((np.min(X), np.max(X)))
        ax2.grid(True)
        return ax1, ax2

    def import_example(self, data='2dpeaks', url=None, sep=';'):
        X = _import_example(data=data, url=url, sep=sep, verbose=self.verbose)
        return X

# %%
def _plot_original(X, xs, labx, min_peaks, max_peaks, title=None, legend=True, ax=None, figsize=(15, 8)):
    uilabx = np.unique(labx)
    uilabx = uilabx[~np.isnan(uilabx)]

    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, X, 'k')
    ax.plot(max_peaks, X[max_peaks], "x", label='Top')
    ax.plot(min_peaks, X[min_peaks], "o", label='Bottom')

    # Color each detected label
    s=np.arange(0, len(X))
    for i in uilabx:
        idx=(labx==i)
        ax.plot(s[idx], X[idx])
        # plt.plot(s[idx], X[idx], label='peak' + str(i))

    if legend: ax.legend(loc=0)
    ax.set_title(title)
    ax.grid(True)
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
        x = [0,   13,  22,  30,  35,  38,   42,   51,   57,   67,  73,   75,  89,   126,  141,  150,  200 ]
        y = [1.5, 0.8, 1.2, 0.2, 0.4, 0.39, 0.42, 0.22, 0.23, 0.1, 0.11, 0.1, 0.14, 0.09, 0.04,  0.02, 0.01]
        X = np.c_[x, y]
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
        cv2 = stats._import_cv2()
        X = cv2.imread(PATH_TO_DATA)
    else:
        X = pd.read_csv(PATH_TO_DATA, sep=sep).values
    # Return
    return X

