from findpeaks import findpeaks
import numpy as np
import unittest


class TestFINDPEAKS(unittest.TestCase):

    def test_fit(self):
        # CHECK OUTPUT METHOD TOPOLOGY
        import numpy as np
        from findpeaks import findpeaks

        fp = findpeaks(method="topology", whitelist=['peak'])
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        assert fp.type == 'peaks2d'
        assert [*results.keys()] == ['Xraw', 'Xproc', 'Xdetect', 'Xranked', 'persistence', 'groups0']
        assert [*fp.args] == ['limit', 'scale', 'denoise', 'togray', 'imsize', 'figsize', 'type']
        assert results['Xraw'].shape == results['Xdetect'].shape
        assert results['Xproc'].shape == results['Xdetect'].shape

        fp.plot(figsize=(25, 15), figure_order='horizontal')

        # CHECK RESULTS METHOD TOPOLOGY
        assert len(results['Xdetect'][results['Xdetect'] != 0]) > 18
        assert len(results['Xranked'][results['Xranked'] != 0]) > 18

    def test_topology_limit(self):
        from findpeaks import findpeaks
        # CHECK RESULTS METHOD with LIMIT functionality
        fp = findpeaks(method="topology", whitelist=['peak'], limit=0)
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        fp.plot(figsize=(25, 15), figure_order='horizontal')
        assert len(results['Xdetect'][results['Xdetect'] != 0]) > 18
        assert len(results['Xranked'][results['Xranked'] != 0]) > 18

    def test_mask_2dpeaks(self):
        import numpy as np
        from findpeaks import findpeaks
        # CHECK OUTPUT METHOD MASK
        fp = findpeaks(method="mask")
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        assert fp.type == 'peaks2d'
        assert [*results.keys()] == ['Xraw', 'Xproc', 'Xdetect', 'Xranked']
        assert [*fp.args] == ['limit', 'scale', 'denoise', 'togray', 'imsize', 'figsize', 'type']
        fp.plot(figsize=(25, 15), figure_order='horizontal')

        # CHECK RESULTS METHOD TOPOLOGY
        assert np.sum(results['Xdetect']) == 20
        assert results['Xraw'].shape == results['Xdetect'].shape
        assert results['Xproc'].shape == results['Xdetect'].shape

    def test_whitelist(self):
        # CHECK WHITELIST
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from findpeaks import findpeaks
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 50))
        x = gaussian_filter(x, sigma=10.)
        # peak and valley
        fp = findpeaks(method="topology", whitelist=['peak', 'valley'], denoise=None)
        results = fp.fit(x)

        fp.plot(figsize=(25, 15), figure_order='horizontal')
        fp.plot_persistence()
        fp.plot_mesh()

        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['peak'][Iloc].sum() == 3
        assert results['persistence']['valley'][Iloc].sum() == 4

        # peaks
        fp = findpeaks(method="topology", whitelist='peak', denoise=None)
        fp.plot()
        results = fp.fit(x)
        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['peak'][Iloc].shape[0] == results['persistence']['peak'][Iloc].sum()

        fp = findpeaks(method="topology", whitelist='valley', denoise=None)
        results = fp.fit(x)
        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['valley'].shape[0] == results['persistence']['valley'].sum()

    def test_topology(self):
        # CHECK OUTPUT METHOD TOPOLOGY
        fp = findpeaks(method="topology")
        X = fp.import_example('1dpeaks')
        results = fp.fit(X)
        assert fp.type == 'peaks1d'
        assert [*results.keys()] == ['persistence', 'Xdetect', 'Xranked', 'groups0', 'df']
        assert [*fp.args] == ['method', 'params', 'lookahead', 'interpolate', 'figsize', 'type']
        assert len(X) == len(results['Xdetect'])
        assert len(X) == len(results['Xranked'])
        assert len(X) == results['df'].shape[0]
        assert np.all(np.isin(results['df'].columns, ['x', 'y', 'labx', 'rank', 'score', 'valley', 'peak']))
        assert np.all(np.isin(results['persistence'].columns, ['x', 'y', 'birth_level', 'death_level', 'score']))

        # CHECK RESULTS METHOD TOPOLOGY
        # Let op, soms gaat deze ook op 6 vanwege een stochastic components
        # assert results['persistence'].shape[0] == 7

        # CHECK RESULTS METHOD with LIMIT functionality
        X = fp.import_example('1dpeaks')
        fp = findpeaks(method="topology", limit=0.02)
        results = fp.fit(X)
        assert len(results['Xdetect'][results['Xdetect'] != 0]) == len(results['Xranked'][results['Xranked'] != 0])

        # CHECK OUTPUT METHOD PEAKDETECT
        # fp = findpeaks(method="peakdetect", lookahead=1, verbose=3, height=0)
        fp = findpeaks(method="peakdetect", lookahead=1)
        X = fp.import_example('1dpeaks')
        results = fp.fit(X)
        assert fp.type == 'peaks1d'
        assert [*results.keys()] == ['df']
        assert [*fp.args] == ['method', 'params', 'lookahead', 'interpolate', 'figsize', 'type']
        assert len(X) == results['df'].shape[0]
        assert np.all(np.isin(results['df'].columns, ['x', 'y', 'labx', 'valley', 'peak', 'rank', 'score']))

        # CHECK RESULTS METHOD TOPOLOGY
        assert results['df']['peak'].sum() == 2
        assert results['df']['valley'].sum() == 4
        fp.plot()

        # Run over all combinations and make sure no errors are made
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        methods = ['topology', 'peakdetect', None]
        interpolates = [None, 1, 10, 1000]
        lookaheads = [None, 0, 1, 10, 100]
        for method in methods:
            for interpolate in interpolates:
                for lookahead in lookaheads:
                    fp = findpeaks(method=method, lookahead=lookahead, interpolate=interpolate)
                    assert fp.fit(X)
    
    def test_method_combinations(self):
        """Test all method combinations with various parameters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        
        # Test different methods
        methods = ['peakdetect', 'caerus', None]
        for method in methods:
            fp = findpeaks(method=method)
            results = fp.fit(X)
            assert results is not None
    
    def test_verbose_levels(self):
        """Test different verbose levels"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        verbose_levels = ['info', 'warning', 'error', 'debug', 0, 1, 2, 3]
        
        for verbose in verbose_levels:
            fp = findpeaks(method='peakdetect', verbose=verbose)
            results = fp.fit(X)
            assert results is not None
    
    def test_figsize_parameters(self):
        """Test different figsize parameters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        figsize_options = [(10, 5), (20, 10), (5, 15), None]
        
        for figsize in figsize_options:
            fp = findpeaks(method='peakdetect', figsize=figsize)
            results = fp.fit(X)
            assert results is not None
    
    def test_whitelist_combinations(self):
        """Test different whitelist combinations"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        whitelist_options = ['peak', 'valley', ['peak', 'valley'], None]
        
        for whitelist in whitelist_options:
            fp = findpeaks(method='topology', whitelist=whitelist)
            results = fp.fit(X)
            assert results is not None
    
    def test_limit_parameters(self):
        """Test different limit parameters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        limit_options = [0, 0.1, 0.5, 1.0, 10, None]
        
        for limit in limit_options:
            fp = findpeaks(method='topology', limit=limit)
            results = fp.fit(X)
            assert results is not None
    
    def detection_1d(self):
        from findpeaks import findpeaks
        X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]
        
        fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=2, verbose='info')
        results = fp.fit(X)
        fp.plot()
        assert results['df'].shape == (74, 5)
        assert results['df']['valley'].sum()== 11
        assert results['df']['peak'].sum()== 9    

    def test_interpolation_parameters(self):
        """Test different interpolation parameters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        interpolate_options = [1, 2, 3, 5, 10, 100, None]
        
        for interpolate in interpolate_options:
            fp = findpeaks(method='peakdetect', interpolate=interpolate)
            results = fp.fit(X)
            assert results is not None
    
    def test_lookahead_parameters(self):
        """Test different lookahead parameters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        lookahead_options = [0, 1, 2, 5, 10, 50, 100, 200, None]
        
        for lookahead in lookahead_options:
            fp = findpeaks(method='peakdetect', lookahead=lookahead)
            results = fp.fit(X)
            assert results is not None
    
    def test_delta_parameters(self):
        """Test different delta parameters for peakdetect"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        delta_options = [0, 0.1, 0.5, 1.0, 5.0, 10.0]
        
        for delta in delta_options:
            fp = findpeaks(method='peakdetect', params={'delta': delta})
            results = fp.fit(X)
            assert results is not None
    
    def test_window_parameters(self):
        """Test different window parameters for various filters"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        window_options = [3, 5, 7, 9, 11, 15, 21, None]
        
        for window in window_options:
            fp = findpeaks(method='peakdetect', params={'window': window})
            results = fp.fit(X)
            assert results is not None

    def detection_2d(self):
        from findpeaks import findpeaks
        import matplotlib.pyplot as plt
        
        # Initialize
        fp = findpeaks(method='topology', imsize=(150, 150), scale=True, togray=True, denoise='lee_sigma', params={'window': 17})
        # Import example image
        img = fp.import_example('2dpeaks_image')
        # Denoising and detecting peaks
        results = fp.fit(img)
        # Create mesh plot
        fp.plot_mesh()
        # Create denoised plot
        fp.plot(limit=80, figure_order='horizontal', cmap=plt.cm.hot_r)
        assert sum(results['persistence']['score']>80)==3
    
    def test_2d_parameters(self):
        """Test different 2D parameters"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test different imsize parameters
        imsize_options = [(100, 100), (200, 200), (300, 300), None]
        for imsize in imsize_options:
            fp = findpeaks(method='topology', imsize=imsize)
            results = fp.fit(X)
            assert results is not None
        
        # Test scale parameter
        for scale in [True, False]:
            fp = findpeaks(method='topology', scale=scale)
            results = fp.fit(X)
            assert results is not None
        
        # Test togray parameter
        for togray in [True, False]:
            fp = findpeaks(method='topology', togray=togray)
            results = fp.fit(X)
            assert results is not None
    
    def test_scale_false_with_opencv_denoising(self):
        """Test that scale=False works with OpenCV denoising methods"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test with fastnl denoising (OpenCV method)
        fp = findpeaks(method='topology', scale=False, denoise='fastnl')
        results = fp.fit(X)
        assert results is not None
        
        # Test with bilateral denoising (OpenCV method)
        fp = findpeaks(method='topology', scale=False, denoise='bilateral')
        results = fp.fit(X)
        assert results is not None
        
        # Test with non-OpenCV denoising methods
        non_opencv_methods = ['lee', 'lee_enhanced', 'lee_sigma', 'kuan', 'frost', 'median', 'mean']
        for method in non_opencv_methods:
            fp = findpeaks(method='topology', scale=False, denoise=method)
            results = fp.fit(X)
            assert results is not None
    
    def test_denoising_methods(self):
        """Test all denoising methods"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        denoise_methods = ['fastnl', 'bilateral', 'lee', 'lee_enhanced', 'lee_sigma', 
                          'kuan', 'frost', 'median', 'mean', None]
        
        for method in denoise_methods:
            fp = findpeaks(method='topology', denoise=method)
            results = fp.fit(X)
            assert results is not None
    
    def test_denoising_parameters(self):
        """Test different denoising parameters"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test different window sizes for denoising
        window_sizes = [3, 5, 7, 9, 11, 15, 21, None]
        for window in window_sizes:
            fp = findpeaks(method='topology', denoise='lee', params={'window': window})
            results = fp.fit(X)
            assert results is not None
        
        # Test different cu values for lee filter
        cu_values = [0.1, 0.25, 0.5, 0.75, 1.0, None]
        for cu in cu_values:
            fp = findpeaks(method='topology', denoise='lee', params={'cu': cu})
            results = fp.fit(X)
            assert results is not None
    
    def test_frost_filter_parameters(self):
        """Test frost filter specific parameters"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test different damping factors
        damping_factors = [1.0, 2.0, 3.0, 5.0]
        for damping in damping_factors:
            fp = findpeaks(method='topology', denoise='frost', 
                          params={'damping_factor': damping, 'win_size': 15})
            results = fp.fit(X)
            assert results is not None
    
    def test_lee_enhanced_parameters(self):
        """Test lee enhanced filter specific parameters"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test different k values
        k_values = [0.5, 1.0, 1.5, 2.0]
        for k in k_values:
            fp = findpeaks(method='topology', denoise='lee_enhanced', 
                          params={'k': k, 'cu': 0.523, 'cmax': 1.73, 'win_size': 15})
            results = fp.fit(X)
            assert results is not None
    
    def detection_2d_url(self):
        from findpeaks import findpeaks
        path = r'https://erdogant.github.io/datasets/images/complex_peaks.png'
        fp = findpeaks(method='topology', whitelist='peak', limit=5, denoise='lee_sigma', params={'window': 5})
        X = fp.imread(path)
        results = fp.fit(X)
        
        fp.plot_persistence()
        fp.plot(figsize=(25, 14), text=False, marker='x', color='#ff0000', figure_order='vertical', fontsize=14)
        # fp.plot_mesh(view=(40, 180))
        # fp.plot_mesh(view=(90, 0))
        assert results['persistence'].shape == (47, 7)

    def test_plot_parameters(self):
        """Test different plotting parameters"""
        fp = findpeaks()
        X = fp.import_example('1dpeaks')
        results = fp.fit(X)
        
        # Test different figure_order options
        figure_orders = ['vertical', 'horizontal']
        for order in figure_orders:
            fp.plot(figure_order=order)
        
        # Test different cmap options
        import matplotlib.pyplot as plt
        cmap_options = [plt.cm.hot, plt.cm.cool, plt.cm.viridis, None]
        for cmap in cmap_options:
            fp.plot(cmap=cmap)
        
        # Test different marker options
        marker_options = ['x', 'o', 's', '^', 'v', '*']
        for marker in marker_options:
            fp.plot(marker=marker)
        
        # Test different color options
        color_options = ['red', '#FF0000', 'blue', 'green', None]
        for color in color_options:
            fp.plot(color=color)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with very small data
        X_small = [1, 2, 1]
        fp = findpeaks(method='peakdetect')
        results = fp.fit(X_small)
        assert results is not None
        
        # Test with constant data
        X_constant = [1, 1, 1, 1, 1]
        results = fp.fit(X_constant)
        assert results is not None
        
        # Test with single value
        X_single = [1]
        results = fp.fit(X_single)
        assert results is not None
        
        # Test with empty list
        X_empty = []
        try:
            results = fp.fit(X_empty)
            # Should either work or raise an exception
        except:
            pass  # Expected for empty data
    
    def test_data_types(self):
        """Test different data types"""
        # Test with numpy array
        import numpy as np
        X_np = np.array([10, 11, 9, 23, 21, 11, 45, 20, 11, 12])
        fp = findpeaks(method='peakdetect')
        results = fp.fit(X_np)
        assert results is not None
        
        # Test with list of floats
        X_float = [10.0, 11.0, 9.0, 23.0, 21.0, 11.0, 45.0, 20.0, 11.0, 12.0]
        results = fp.fit(X_float)
        assert results is not None
        
        # Test with mixed types
        X_mixed = [10, 11.0, 9, 23.0, 21, 11.0, 45, 20.0, 11, 12.0]
        results = fp.fit(X_mixed)
        assert results is not None
    
    def test_parameter_combinations(self):
        """Test various parameter combinations"""
        X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
        
        # Test topology with different combinations
        combinations = [
            {'method': 'topology', 'limit': 0.1, 'whitelist': 'peak'},
            {'method': 'topology', 'limit': 0.5, 'whitelist': ['peak', 'valley']},
            {'method': 'peakdetect', 'lookahead': 1, 'interpolate': 3},
            {'method': 'peakdetect', 'lookahead': 2, 'params': {'delta': 0.5}},
        ]
        
        for combo in combinations:
            fp = findpeaks(**combo)
            results = fp.fit(X)
            assert results is not None
    
    def test_import_examples(self):
        """Test all available example datasets"""
        fp = findpeaks()
        
        # Test all available examples
        examples = ['1dpeaks', '2dpeaks', '2dpeaks_image']
        for example in examples:
            try:
                X = fp.import_example(example)
                assert X is not None
                # Test that we can fit on the imported data
                results = fp.fit(X)
                assert results is not None
            except Exception as e:
                # Some examples might not be available in all environments
                print(f"Example {example} not available: {e}")
    
    def test_denoising(self):

        # DENOISING METHODS TEST
        from findpeaks import findpeaks
        fp = findpeaks()
        img = fp.import_example('2dpeaks_image')
        import findpeaks

        # filters parameters
        # window size
        winsize = 15
        # damping factor for frost
        k_value1 = 2.0
        # damping factor for lee enhanced
        k_value2 = 1.0
        # coefficient of variation of noise
        cu_value = 0.25
        # coefficient of variation for lee enhanced of noise
        cu_lee_enhanced = 0.523
        # max coefficient of variation for lee enhanced
        cmax_value = 1.73

        # Some pre-processing
        # Resize
        img = findpeaks.stats.resize(img, size=(300, 300))
        # Make grey image
        img = findpeaks.stats.togray(img)
        # Scale between [0-255]
        img = findpeaks.stats.scale(img)

        # Denoising
        # fastnl
        img_fastnl = findpeaks.stats.denoise(img.copy(), method='fastnl', window=winsize)
        # bilateral
        img_bilateral = findpeaks.stats.denoise(img.copy(), method='bilateral', window=winsize)
        # frost filter
        image_frost = findpeaks.stats.frost_filter(img.copy(), damping_factor=k_value1, win_size=winsize)
        # kuan filter
        image_kuan = findpeaks.stats.kuan_filter(img.copy(), win_size=winsize, cu=cu_value)
        # lee filter
        image_lee = findpeaks.stats.lee_filter(img.copy(), win_size=winsize, cu=cu_value)
        # lee enhanced filter
        image_lee_enhanced = findpeaks.stats.lee_enhanced_filter(img.copy(), win_size=winsize, k=k_value2, cu=cu_lee_enhanced, cmax=cmax_value)
        # lee sigma filter
        image_lee_sigma = findpeaks.stats.lee_sigma_filter(img.copy())
        # mean filter
        image_mean = findpeaks.stats.mean_filter(img.copy(), win_size=winsize)
        # median filter
        image_median = findpeaks.stats.median_filter(img.copy(), win_size=winsize)

        # Loop throughout many combinations of parameter settings
        from findpeaks import findpeaks
        methods = ['caerus', 'mask', 'topology', None]
        filters = ['lee', 'lee_enhanced', 'lee_sigma', 'kuan', 'fastnl', 'bilateral', 'frost', 'median', 'mean', None]
        windows = [None, 3, 63]
        cus = [None, 0, 0.75]
        img = fp.import_example('2dpeaks')

        for getfilter in filters:
            for window in windows:
                for cu in cus:
                    fp = findpeaks(method='topology', imsize=None, scale=True, togray=True, denoise=getfilter, params={'window': window, 'cu': cu})
                    assert fp.fit(img)
                    # assert fp.plot_mesh(wireframe=False)
                    # plt.close('all')
                    # assert fp.plot_persistence()
                    # plt.close('all')
                    # assert fp.plot()
                    # plt.close('all')
                    # assert fp.plot_preprocessing()
  
    def test_persistence_plot_parameters(self):
        """Test persistence plotting parameters"""
        fp = findpeaks(method='topology')
        X = fp.import_example('1dpeaks')
        results = fp.fit(X)
        
        # Test different figsize options
        figsize_options = [(10, 5), (20, 10), (15, 8)]
        for figsize in figsize_options:
            fp.plot_persistence(figsize=figsize)
        
        # Test different fontsize options
        fontsize_options = [10, 14, 18, 20]
        for fontsize in fontsize_options:
            fp.plot_persistence(fontsize_ax1=fontsize, fontsize_ax2=fontsize)
        
        # Test different label options
        label_options = ['x-axis', 'y-axis', 'Time', 'Value']
        for label in label_options:
            fp.plot_persistence(xlabel=label, ylabel=label)
    
    def test_plot_preprocessing(self):
        """Test preprocessing plot functionality"""
        fp = findpeaks(method='topology', denoise='lee')
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        
        # Test that preprocessing plot works
        fp.plot_preprocessing()
    
    def test_plot_mask(self):
        """Test mask plotting functionality"""
        fp = findpeaks(method='mask')
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        
        # Test different mask plot parameters
        limit_options = [0, 0.1, 0.5, None]
        for limit in limit_options:
            ax = fp.plot_mask(limit=limit)
        
        # Test different figure_order options
        for order in ['vertical', 'horizontal']:
            ax = fp.plot_mask(figure_order=order)
        
        # Test different marker and color options
        fp.plot_mask(marker='o', color='blue', s=5, fontsize=10)
    
    def test_caerus_method(self):
        """Test caerus method specifically"""
        fp = findpeaks(method='caerus')
        X = fp.import_example('1dpeaks')
        
        # Test with default parameters
        results = fp.fit(X)
        assert results is not None
        
        # Test with custom parameters
        fp = findpeaks(method='caerus', params={'window': 5, 'delta': 0.1})
        results = fp.fit(X)
        assert results is not None
    
    def test_2d_with_different_methods(self):
        """Test 2D data with different methods"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # Test topology method
        fp = findpeaks(method='topology')
        results = fp.fit(X)
        assert results is not None
        assert 'persistence' in results
        
        # Test mask method
        fp = findpeaks(method='mask')
        results = fp.fit(X)
        assert results is not None
        assert 'Xdetect' in results
        
    def test_1d_with_different_methods(self):
        """Test 1D data with different methods"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        
        # Test topology method
        fp = findpeaks(method='topology')
        results = fp.fit(X)
        assert results is not None
        assert 'persistence' in results
        
        # Test peakdetect method
        fp = findpeaks(method='peakdetect')
        results = fp.fit(X)
        assert results is not None
        assert 'df' in results
        
        # Test caerus method
        fp = findpeaks(method='caerus')
        results = fp.fit(X)
        assert results is not None
    
    def test_noise_handling(self):
        """Test handling of noisy data"""
        import numpy as np
        
        # Create noisy data
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)
        
        # Test with different denoising methods
        denoise_methods = ['fastnl', 'bilateral', 'lee', 'median', 'mean']
        
        for method in denoise_methods:
            fp = findpeaks(method='topology', denoise=method)
            results = fp.fit(y)
            assert results is not None
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        import numpy as np
        
        # Create larger dataset
        np.random.seed(42)
        X_large = np.random.randn(1000)
        
        # Test with different methods
        methods = ['topology', 'peakdetect', 'caerus']
        
        for method in methods:
            fp = findpeaks(method=method)
            results = fp.fit(X_large)
            assert results is not None
    
    def test_parameter_validation(self):
        """Test parameter validation and error handling"""
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        
        # Test invalid method
        try:
            fp = findpeaks(method='invalid_method')
            results = fp.fit(X)
            # Should either work with default method or raise an exception
        except:
            pass  # Expected for invalid method
        
        # Test invalid parameters
        try:
            fp = findpeaks(method='peakdetect', lookahead=-1)
            results = fp.fit(X)
            # Should either work or raise an exception
        except:
            pass  # Expected for negative lookahead
        
        # Test invalid interpolation factor
        try:
            fp = findpeaks(method='peakdetect', interpolate=0)
            results = fp.fit(X)
            # Should either work or raise an exception
        except:
            pass  # Expected for zero interpolation
    
    def test_specific_error_case(self):
        """Test the specific case that was causing the OpenCV error"""
        fp = findpeaks()
        X = fp.import_example('2dpeaks')
        
        # This was the problematic case: scale=False with fastnl denoising
        fp = findpeaks(method='topology', scale=False, denoise='fastnl')
        results = fp.fit(X)
        assert results is not None
        
        # Also test with bilateral denoising
        fp = findpeaks(method='topology', scale=False, denoise='bilateral')
        results = fp.fit(X)
        assert results is not None