# %%
# import os
# os.chdir(os.path.dirname(os.path.abspath('examples.py')))
# import findpeaks
# print(dir(findpeaks))
# print(findpeaks.__version__)

# pip install opencv-python
import matplotlib.pyplot as plt
# from findpeaks import findpeaks

# %% 
# =============================================================================
# Issue with numpy > 1.26.4
# =============================================================================

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

assert len(results['Xdetect'][results['Xdetect'] != 0]) > 18
assert len(results['Xranked'][results['Xranked'] != 0]) > 18


# %%
import numpy as np

# Simulate x-axis
x = np.linspace(0, 100, 200)
# Create the primary peak (sharp and high)
peak1 = np.exp(-(x - 30)**2 / (2 * 3**2)) * 50
# Add a few smaller bumps
peak2 = np.exp(-(x - 70)**2 / (2 * 5**2)) * 20
peak3 = np.exp(-(x - 80)**2 / (2 * 3**2)) * 15
# Combine and add some randomness
y = peak1 + peak2 + peak3 + np.random.normal(0, 0.5, size=len(x))

from findpeaks import findpeaks
fp = findpeaks(method="peakdetect", lookahead=100, interpolate=10)
results = fp.fit(y)
ax = fp.plot()


# %%
from findpeaks import findpeaks
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(method="topology", lookahead=1, verbose='info')
results = fp.fit(X)
ax = fp.plot()
ax = fp.plot_persistence()

# fp.check_logger(verbose='info')

#%%
# Import library
from findpeaks import findpeaks
# Initialize
fp = findpeaks(method='topology')
# Example 1d-vector
X = fp.import_example('1dpeaks')
# Plot
plt.plot(X); plt.grid(True)

# Fit topology method on the 1d-vector
results = fp.fit(X)
# Plot the results
ax = fp.plot_persistence()


# %%
from findpeaks import findpeaks
#smth = your dummy data here

# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# Initialize
fp = findpeaks(method='peakdetect', lookahead=1)
results = fp.fit(X)
# Plot
fp.plot()

fp = findpeaks(method='topology', lookahead=1)
results = fp.fit(X)
fp.plot()
fp.plot_persistence()


# %%
# Import library
from findpeaks import findpeaks
# Initialize findpeaks with cearus method.
# The default setting is that it only return peaks-vallyes with at least 5% difference. We can change this using params
# fp = findpeaks(method='caerus',  params={'minperc': 10, 'window': 50})
fp = findpeaks(method='caerus')
# Import example data
X = fp.import_example('facebook')
# Fit
results = fp.fit(X)
# Make the plot
fp.plot()

# %%
from findpeaks import findpeaks
# Data
X = [1153,672,501,1068,1110,574,135,23,3,47,252,812,1182]
# Initialize
fp = findpeaks(lookahead=1)
results = fp.fit(X)
# Plot
fp.plot()
results.get('df')


# %% Issue29

# Import library
import findpeaks
# Small dataset
X = [10,11,9,23,21,11,45,20,11,12]
# Interpolate the data using linear by factor 10
Xi = findpeaks.interpolate.interpolate_line1d(X, method='linear', n=10, showfig=True)
# Print message
print('Input data lenth: %s, interpolated length: %s' %(len(X), len(Xi)))
# Input data lenth: 10, interpolated length: 100

# Load library
from findpeaks import findpeaks
# Data
X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
# Initialize
fp = findpeaks(lookahead=1, interpolate=1)
results = fp.fit(X)
# Plot
fp.plot()


# %% Issue
# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# # Initialize
# fp = findpeaks(method='peakdetect', lookahead=1)
# results = fp.fit(X)
# # Plot
# fp.plot()

fp = findpeaks(method='topology', lookahead=1)
results = fp.fit(X)
# fp.plot()
fp.plot_persistence()


# %% find peak and valleys in 2d images.
import numpy as np
from scipy.ndimage import gaussian_filter
from findpeaks import findpeaks
rng = np.random.default_rng(42)
x = rng.normal(size=(50, 50))
x = gaussian_filter(x, sigma=10)
# peak and valley
fp = findpeaks(method="topology", whitelist=['peak', 'valley'], denoise=None, verbose='info')
results = fp.fit(x)
results['persistence']

fp.plot(figsize=(25, 15), figure_order='horizontal', cmap='hot_r', text=True)
fp.plot_persistence()
fp.plot_mesh()

# %%
import time
from findpeaks import findpeaks
path = r'https://erdogant.github.io/datasets/images/complex_peaks.png'
fp = findpeaks(method='topology', whitelist='peak', limit=5, denoise='lee_sigma', params={'window': 5})
X = fp.imread(path)

start_orig = time.time()
results = fp.fit(X)
time_spend = time.time() - start_orig

# result_df = results['persistence']
# peak = result_df.index[result_df['peak']==True].tolist()

fp.plot_persistence()
fp.plot(figsize=(25, 14), text=False, marker='x', color='#ff0000', figure_order='vertical', fontsize=14)
# fp.plot_mesh(view=(40, 180))
# fp.plot_mesh(view=(90, 0))
assert results['persistence'].shape == (47, 7)

# %%
from findpeaks import findpeaks
path = r'https://erdogant.github.io/datasets/images/complex_peaks.png'
fp = findpeaks(method='topology', whitelist='peak', denoise='lee_enhanced', params={'window': 5}, verbose='debug')
X = fp.imread(path)
results = fp.fit(X)
ax = fp.plot_persistence()
ax = fp.plot(text=False)
# fp.plot_mesh()

fp.results['persistence'].iloc[0:10,:]

# %%
# Import library
from findpeaks import findpeaks

# Initialize
fp = findpeaks(method='topology',
               scale=True,
               togray=True,
               imsize=(150, 150),
               denoise='lee_sigma',
               params={'window': 17},
               limit=160,
               )

# Import example image
img = fp.import_example('2dpeaks_image')

# Denoising and detecting peaks
results = fp.fit(img)
# Create mesh plot
fp.plot_mesh()
# Create denoised plot
fp.plot(figure_order='horizontal')
fp.plot_persistence()

# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=1, params={'delta': 1}, verbose='info')
# fp = findpeaks(method='topology')
results = fp.fit(X)
fp.plot()


# %% New functionality:
import findpeaks as fp
import matplotlib.pyplot as plt

# Import example
img = fp.import_example('2dpeaks_image')
# Resize
img = fp.stats.resize(img, size=(150, 150))
# Make grey image
img = fp.stats.togray(img)
# Scale between [0-255]
img = fp.stats.scale(img)
# Filter
img_filtered = fp.stats.lee_sigma_filter(img, win_size=17)

plt.figure()
fig, axs = plt.subplots(1,2)
axs[0].imshow(img, cmap='gray'); axs[0].set_title('Input')
axs[1].imshow(img_filtered, cmap='gray'); axs[1].set_title('Lee sigma filter')


# %%
# Import library
from findpeaks import findpeaks
import matplotlib.pyplot as plt

# Initialize
fp = findpeaks(method='topology', imsize=(150, 150), scale=True, togray=True, denoise='lee_sigma',
               params={'window': 17})

# Import example image
img = fp.import_example('2dpeaks_image')

# Denoising and detecting peaks
results = fp.fit(img)
# Create mesh plot
fp.plot_mesh()
# Create denoised plot
fp.plot(limit=80, figure_order='horizontal', cmap=plt.cm.hot_r)

sum(results['persistence']['score']>80)

# %% Issue 18:
from findpeaks import findpeaks
    
fp = findpeaks(method='topology', imsize=False, scale=False, togray=False, denoise=None, params={'window': 15})
X = fp.import_example('2dpeaks')
fp.fit(X)
# fp.plot_mesh(wireframe=False, title='Test', cmap='RdBu', view=(70,5))
fp.plot_mesh()
fp.plot_mesh(xlim=[10, 30], ylim=[4, 10])
fp.plot_mesh(xlim=[10, None], ylim=[4, 10])
fp.plot_mesh(xlim=[10, None], ylim=[4, None])
fp.plot_mesh(xlim=[10, 30], ylim=[4, 10], zlim=[0, 3])
fp.plot_mesh(xlim=[10, 30], ylim=[4, 10], zlim=[0, None])
fp.plot_mesh(xlim=[10, 30], ylim=[4, 10], zlim=[None, 6])
fp.plot_mesh(xlim=[10, 30], ylim=[4, 10], zlim=[2, 6])




# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=1, verbose='info')
results = fp.fit(X)
fp.plot()
assert results['df'].shape == (74, 5)
assert results['df']['valley'].sum()== 13
assert results['df']['peak'].sum()== 12

fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=1, interpolate=10)
results = fp.fit(X)
fp.plot()

# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(method='topology', whitelist=['valley'])
results=fp.fit(X)
fp.plot(xlabel='x', ylabel='y')
fp.plot_persistence(xlabel='x', ylabel='y')

# %% issue in mail
# Import library
from findpeaks import findpeaks
# Initialize
fp = findpeaks(method='topology')
# Example 1d-vector
X = fp.import_example('1dpeaks')

# Fit topology method on the 1d-vector
results = fp.fit(X)
# Plot the results
fp.plot_persistence(fontsize_ax1=12, fontsize_ax2=14)


# %% issue #12
# https://github.com/erdogant/findpeaks/issues/12
import numpy as np

X = np.sin(np.linspace(0, 1, 100))
from findpeaks import findpeaks
# fp = findpeaks(method='caerus', params_caerus={'minperc': 5, 'window': 50})
fp = findpeaks(method='caerus', params={'minperc': 5, 'window': 50})
results = fp.fit(X)
fp.plot()

# %% Issue 13
# https://github.com/erdogant/findpeaks/issues/13
from findpeaks import findpeaks
fp = findpeaks(method="mask", limit=None, denoise=None, params={'window': 3}, verbose=0)
X = fp.import_example("2dpeaks_image")
results = fp.fit(X)
fp.plot(figure_order='horizontal')


# %%
# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# Initialize
fp = findpeaks(method='topology', whitelist=['valley', 'peak'], lookahead=1, verbose=0)
results = fp.fit(X)
fp.plot()
results['df']


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from findpeaks import findpeaks
rng = np.random.default_rng(42)

x = rng.normal(size=(50, 50))
x = gaussian_filter(x, sigma=10.)

# Search for peaks/valleys with a minimum value of 0
fp = findpeaks(method="topology", limit=0, denoise=None, verbose='info')
results = fp.fit(x)

results['persistence']
fp.plot(cmap='coolwarm')

# Plot
plt.imshow(x, cmap="coolwarm", interpolation="none", vmin=0, vmax=255)

fp.plot(cmap='coolwarm')
fp.plot_persistence()
fp.plot_mesh()

plt.imshow(x, cmap="coolwarm", interpolation="none", vmin=0, vmax=255); plt.grid(True)
plt.imshow(fp.results['Xdetect'], cmap='gray_r'); plt.grid(True)
results["persistence"]

results.keys()
results['persistence']
results['Xdetect']

# %%
from findpeaks import findpeaks
verbose='info'

fp = findpeaks(verbose=verbose)
# Import example
X = fp.import_example("btc")

# Make fit
fp = findpeaks(method="topology", verbose=verbose)
results = fp.fit(X)
fp.plot()
fp.plot_persistence(fontsize_ax1=4)

fp = findpeaks(method="peakdetect", lookahead=15, verbose=verbose)
# Make fit
results = fp.fit(X)
fp.plot()

fp = findpeaks(method="caerus", interpolate=None, params={'minperc': 50, 'window':50}, verbose=verbose)
# Make fit
results = fp.fit(X)
ax = fp.plot()


# %%
from findpeaks import findpeaks
import numpy as np

# np.random.seed(100)
np.random.seed(200)
X = np.random.randint(200, size=400)

fp = findpeaks(method='topology', lookahead=1, interpolate=10, verbose='info')
results = fp.fit(X)

fig=fp.plot()
fp.plot_mesh()

# %%
from findpeaks import findpeaks
fp = findpeaks(method="topology", verbose='info', limit=1)
X = fp.import_example("2dpeaks")
results = fp.fit(X)
fp.plot(figure_order='horizontal')

# %%
# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# Initialize
fp = findpeaks(method='peakdetect', lookahead=1, verbose='info')
results = fp.fit(X)
# Plot
fig=fp.plot()

# %%
# Import library
from findpeaks import findpeaks
# Import image example
img = fp.import_example('2dpeaks_image')
# Initializatie
fp = findpeaks(whitelist=['peak', 'valley'], imsize=(300, 300), scale=True, togray=True, denoise='fastnl',
               params={'window': 31}, verbose='info')
# Fit
fp.fit(img)
fp.plot()
fp.results["persistence"]

# Take the minimum score for the top peaks off the diagonal.
limit = fp.results['persistence'][0:5]['score'].min()
fp.plot(text=True, limit=limit)

fp = findpeaks(whitelist=['peak', 'valley'], limit=limit, imsize=(300, 300), scale=True, togray=True, denoise='fastnl',
               params={'window': 31}, verbose='info')
fp.fit(img)

fp.results["persistence"]
fp.plot(text=True)

# Plot
fp.plot_mesh()
# Rotate to make a top view
fp.plot_mesh(view=(90,0))

# %%
from findpeaks import findpeaks
fp = findpeaks(method="topology", limit=None, denoise=None, params={'window': 3}, verbose=0)
X = fp.import_example("2dpeaks_image")
# X = fp.import_example("2dpeaks")
results = fp.fit(X)

fp.plot_persistence()

results["persistence"]

fp.plot()
fp.plot_persistence()
fp.plot_mesh()


fp = findpeaks(method="mask")
X = fp.import_example()
results = fp.fit(X)

fp.plot()
fp.plot_preprocessing()
fp.plot_persistence()
fp.plot_mesh()

# %%
from findpeaks import findpeaks
# X = fp.import_example('1dpeaks')
X = [10,11,9,23,21,11,45,20,11,12]
methods = ['topology', 'peakdetect', None]
interpolates = [None, 1, 10, 1000]
lookaheads =[None, 0, 1, 10, 100]

for method in methods:
    for interpolate in interpolates:
        for lookahead in lookaheads:
            fp = findpeaks(method=method, lookahead=lookahead, interpolate=interpolate)
            results = fp.fit(X)
            # fp.plot()
            # fp.plot_persistence()

# fp.results['df_interp']
fp.results['df']




# %% Run over all methods and many parameters
from findpeaks import findpeaks
savepath=''
methods = ['mask','topology', None]
filters = ['fastnl','bilateral','frost','median','mean', None]
windows = [3, 9, 15, 31, 63]
cus = [0.25, 0.5, 0.75]
verbose='info'

for getfilter in filters:
    for window in windows:
            fp = findpeaks(method='topology', imsize=(300, 300), scale=True, togray=True, denoise=getfilter,
                           params={'window': window}, verbose=verbose)
            img = fp.import_example('2dpeaks_image')
            results = fp.fit(img)
            title = 'Method=' + str(getfilter) + ', window='+str(window)
            _, ax1 = fp.plot_mesh(wireframe=False, title=title, savepath=savepath+title+'.png')

filters = ['lee','lee_enhanced','kuan']
for getfilter in filters:
    for window in windows:
        for cu in cus:
            fp = findpeaks(method='topology', imsize=(300, 300), scale=True, togray=True, denoise=getfilter,
                           params={'window': window, 'cu': cu}, verbose=verbose)
            img = fp.import_example('2dpeaks_image')
            results = fp.fit(img)
            title = 'Method=' + str(getfilter) + ', window='+str(window) + ', cu='+str(cu)
            _, ax1 = fp.plot_mesh(wireframe=False, title=title, savepath=savepath+title+'.png')


#%% Plot each seperately
fp.plot_preprocessing()
fp.plot()
fp.plot_persistence()
fp.plot_mesh()

# Make mesh plot
fp.plot_mesh(view=(0,90))
fp.plot_mesh(view=(90,0))


# %%
from findpeaks import findpeaks

fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10, verbose='info')
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()
fp.plot_persistence()


from findpeaks import findpeaks
fp = findpeaks(method='topology', verbose='info')
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()
fp.plot_persistence()

from findpeaks import findpeaks
fp = findpeaks(method='topology', interpolate=10, verbose='info')
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()
fp.plot_persistence()


from tabulate import tabulate
print(tabulate(fp.results['df'], tablefmt="grid", headers="keys"))
print(tabulate(fp.results['persistence'], tablefmt="grid", headers="keys"))
print(tabulate(fp.results['df_interp'].head(), tablefmt="grid", headers="keys"))

print(tabulate(fp.results['persistence'][0:10], tablefmt="grid", headers="keys"))


# %%
from findpeaks import findpeaks

# 2dpeaks example
fp = findpeaks(method='topology')
img = fp.import_example('2dpeaks')
fp.fit(img)
fp.plot(cmap='hot')
fp.plot()
fp.plot_persistence()

fp = findpeaks(method='mask', verbose='info')
img = fp.import_example()
fp.fit(img)
fp.plot()


# 2dpeaks example with other settings
fp = findpeaks(method='topology', imsize=(300, 300), scale=True, togray=True, denoise='fastnl', params={'window': 31},
               verbose='info')
img = fp.import_example('2dpeaks')
fp.fit(img)
fp.plot()

# %%
from findpeaks import findpeaks
fp = findpeaks(method='topology')
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()

fp.plot_preprocessing()
fp.plot_mesh()
fp.plot_persistence()



# %%
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10)
fp.fit(X)
fp.plot()

fp = findpeaks(method='topology', lookahead=1, interpolate=10)
fp.fit(X)
fp.plot()
fp.plot_persistence()

# %%
from math import pi
import numpy as np
from findpeaks import findpeaks

i = 10000
xs = np.linspace(0,3.7*pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Findpeaks
fp = findpeaks(method='peakdetect', verbose=0)
results=fp.fit(X)
fp.plot()

fp = findpeaks(method='topology', verbose=0)
results=fp.fit(X)

fp.plot_persistence()
# fp.results['Xdetect']>1

# %% Denoising example
import findpeaks
img = findpeaks.import_example('2dpeaks_image')

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
img = findpeaks.stats.resize(img, size=(300,300))
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
# mean filter
image_mean = findpeaks.stats.mean_filter(img.copy(), win_size=winsize)
# median filter
image_median = findpeaks.stats.median_filter(img.copy(), win_size=winsize)

# Plotting
import matplotlib.pyplot as plt
plt.figure(); plt.imshow(img_fastnl, cmap='gray'); plt.title('Fastnl'); plt.grid(False)
plt.figure(); plt.imshow(img_bilateral, cmap='gray'); plt.title('Bilateral')
plt.figure(); plt.imshow(image_frost, cmap='gray'); plt.title('Frost')
plt.figure(); plt.imshow(image_kuan, cmap='gray'); plt.title('Kuan')
plt.figure(); plt.imshow(image_lee, cmap='gray'); plt.title('Lee')
plt.figure(); plt.imshow(image_lee_enhanced, cmap='gray'); plt.title('Lee Enhanced')
plt.figure(); plt.imshow(image_mean, cmap='gray'); plt.title('Mean')
plt.figure(); plt.imshow(image_median, cmap='gray'); plt.title('Median')


from findpeaks import findpeaks
fp = findpeaks(method='topology', imsize=False, scale=False, togray=True, denoise='fastnl', verbose='info')
fp.fit(img)
fp.plot_persistence()
fp.plot_mesh(wireframe=False, title='image_lee_enhanced', view=(30,30))

# %%
