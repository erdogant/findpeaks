# %%
# import os
# os.chdir(os.path.dirname(os.path.abspath('examples.py')))
# import findpeaks
# print(dir(findpeaks))
# print(findpeaks.__version__)

from findpeaks import findpeaks
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(lookahead=1, interpolate=10)
results = fp.fit(X)
fp.plot()

# results['df_interp']


# %% Run over all methods and many parameters
from findpeaks import findpeaks
savepath='./comparison_methods/'
filters = ['fastnl','bilateral','frost','median','mean', None]
windows = [3, 9, 15, 31, 63]
cus = [0.25, 0.5, 0.75]

for getfilter in filters:
    for window in windows:
            fp = findpeaks(mask=0, scale=True, denoise=getfilter, window=window, togray=True, imsize=(300,300), verbose=3)
            img = fp.import_example('2dpeaks_image')
            results = fp.fit(img)
            title = 'Method=' + str(getfilter) + ', window='+str(window)
            _, ax1 = fp.plot_mesh(wireframe=False, title=title, savepath=savepath+title+'.png')

filters = ['lee','lee_enhanced','kuan']
for getfilter in filters:
    for window in windows:
        for cu in cus:
            fp = findpeaks(mask=0, scale=True, denoise=getfilter, window=window, cu=cu, togray=True, imsize=(300,300), verbose=3)
            img = fp.import_example('2dpeaks_image')
            results = fp.fit(img)
            title = 'Method=' + str(getfilter) + ', window='+str(window) + ', cu='+str(cu)
            _, ax1 = fp.plot_mesh(wireframe=False, title=title, savepath=savepath+title+'.png')


#%% Plot each seperately
fp.plot_preprocessing()
fp.plot_mask()
fp.plot_peristence()
fp.plot_mesh()

# Make mesh plot
fp.plot_mesh(view=(0,90))
fp.plot_mesh(view=(90,0))


# %%
from findpeaks import findpeaks
fp = findpeaks()
X = fp.import_example('1dpeaks')

fp = findpeaks(lookahead=1, interpolate=10, verbose=3)
# fp = findpeaks(lookahead=1, interpolate=None, verbose=3)
fp.fit(X[:,1])
fp.plot()
fp.plot(method='topology')
fp.plot_peristence()


# %%
from findpeaks import findpeaks
img = fp.import_example()

# 2dpeaks example
fp = findpeaks()
fp.fit(img)
fp.plot()

# 2dpeaks example with other settings
fp = findpeaks(mask=0, scale=True, denoise='fastnl', window=31, togray=True, imsize=(300,300), verbose=3)
img = fp.import_example('2dpeaks')
fp.fit(img)
fp.plot()

# %%
from findpeaks import findpeaks
fp = findpeaks(mask=0)
X = fp.import_example()
fp.fit(X)
fp.plot()

fp.plot_preprocessing()
fp.plot_mask()
fp.plot_mesh()
fp.plot_peristence()

# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(lookahead=1, verbose=3)
results = fp.fit(X)
fp.plot()
fp.plot(method='topology')
fp.plot_peristence()

fp = findpeaks(lookahead=1, interpolate=10, verbose=3)
results=fp.fit(X)
fp.plot()
fp.plot(method='topology')
fp.plot_peristence()

# %%
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(lookahead=1, interpolate=10)
fp.fit(X)
fp.plot()

# %%
from math import pi
import numpy as np
from findpeaks import findpeaks

i = 10000
xs = np.linspace(0,3.7*pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Findpeaks
fp = findpeaks()
results=fp.fit(X)
fp.plot()
fp.plot(method='topology', legend=False)

# %% Denoising example
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
image_frost = findpeaks.frost_filter(img.copy(), damping_factor=k_value1, win_size=winsize)
# kuan filter
image_kuan = findpeaks.kuan_filter(img.copy(), win_size=winsize, cu=cu_value)
# lee filter
image_lee = findpeaks.lee_filter(img.copy(), win_size=winsize, cu=cu_value)
# lee enhanced filter

# %%
# import cv2
# imageStarsCropped = cv2.bitwise_and(img, img_fastnl)
# plt.figure(); plt.imshow(imageStarsCropped, cmap='gray'); plt.grid(False)
# np.std(img)
# np.std(imageStarsCropped)
# np.std(img_fastnl)
# fp.fit(imageStarsCropped)


image_lee_enhanced = findpeaks.lee_enhanced_filter(img.copy(), win_size=winsize, k=k_value2, cu=cu_lee_enhanced, cmax=cmax_value)
# mean filter
image_mean = findpeaks.mean_filter(img.copy(), win_size=winsize)
# median filter
image_median = findpeaks.median_filter(img.copy(), win_size=winsize)

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
fp = findpeaks(scale=False, denoise=None, togray=False, imsize=False, verbose=3)
fp.fit(image_lee_enhanced)
fp.plot_peristence()
fp.plot_mesh(wireframe=False, title='image_lee_enhanced')
