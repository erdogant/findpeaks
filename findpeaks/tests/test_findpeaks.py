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

        # CHECK RESULTS METHOD with LIMIT functionality
        fp = findpeaks(method="topology", whitelist=['peak'], limit=0)
        X = fp.import_example('2dpeaks')
        results = fp.fit(X)
        fp.plot(figsize=(25, 15), figure_order='horizontal')
        assert len(results['Xdetect'][results['Xdetect'] != 0]) > 18
        assert len(results['Xranked'][results['Xranked'] != 0]) > 18

        # CHECK OUTPUT METHOD MASK
        fp = findpeaks(method="mask", verbose=3)
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

        # CHECK WHITELIST
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from findpeaks import findpeaks
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 50))
        x = gaussian_filter(x, sigma=10.)
        # peak and valley
        fp = findpeaks(method="topology", whitelist=['peak', 'valley'], denoise=None, verbose=3)
        results = fp.fit(x)

        fp.plot(figsize=(25, 15), figure_order='horizontal')
        fp.plot_persistence()
        fp.plot_mesh()

        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['peak'][Iloc].sum() == 3
        assert results['persistence']['valley'][Iloc].sum() == 4

        # peaks
        fp = findpeaks(method="topology", whitelist='peak', denoise=None, verbose=3)
        fp.plot()
        results = fp.fit(x)
        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['peak'][Iloc].shape[0] == results['persistence']['peak'][Iloc].sum()

        fp = findpeaks(method="topology", whitelist='valley', denoise=None, verbose=3)
        results = fp.fit(x)
        Iloc = results['persistence']['score'] > 1
        assert results['persistence']['valley'].shape[0] == results['persistence']['valley'].sum()

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
        assert results['persistence'].shape[0] == 7

        # CHECK RESULTS METHOD with LIMIT functionality
        X = fp.import_example('1dpeaks')
        fp = findpeaks(method="topology", limit=0.02)
        results = fp.fit(X)
        assert len(results['Xdetect'][results['Xdetect'] != 0]) == len(results['Xranked'][results['Xranked'] != 0])

        # CHECK OUTPUT METHOD PEAKDETECT
        fp = findpeaks(method="peakdetect", lookahead=1, verbose=3, height=0)
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

        # Run over all combinations and make sure no errors are made
        X = [10, 11, 9, 23, 21, 11, 45, 20, 11, 12]
        methods = ['topology', 'peakdetect', None]
        interpolates = [None, 1, 10, 1000]
        lookaheads = [None, 0, 1, 10, 100]
        for method in methods:
            for interpolate in interpolates:
                for lookahead in lookaheads:
                    fp = findpeaks(method=method, lookahead=lookahead, interpolate=interpolate, verbose=0)
                    assert fp.fit(X)

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
        image_lee_enhanced = findpeaks.stats.lee_enhanced_filter(img.copy(), win_size=winsize, k=k_value2,
                                                                 cu=cu_lee_enhanced, cmax=cmax_value)
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
                    fp = findpeaks(method='topology', imsize=None, scale=True, togray=True, denoise=getfilter,
                                   params={'window': window, 'cu': cu}, verbose=3)
                    assert fp.fit(img)
                    # assert fp.plot_mesh(wireframe=False)
                    # plt.close('all')
                    # assert fp.plot_persistence()
                    # plt.close('all')
                    # assert fp.plot()
                    # plt.close('all')
                    # assert fp.plot_preprocessing()
