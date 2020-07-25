# %%
# import findpeaks
# print(dir(findpeaks))
# print(findpeaks.__version__)

# %%
from findpeaks import findpeaks

fp = findpeaks(mask=0, scale=True, denoise=None, togray=True, resize=(300,300), verbose=3)
img = fp.import_example('2dpeaks_image')
results = fp.fit(img)
fp.plot()

fp = findpeaks(mask=0, scale=True, denoise='bilateral', h=30, togray=True, resize=(300,300), verbose=3)
img = fp.import_example('2dpeaks_image')
results = fp.fit(img)
fp.plot()

fp = findpeaks(mask=0, scale=True, denoise='fastnl', h=30, togray=True, resize=(300,300), verbose=3)
img = fp.import_example('2dpeaks_image')
results = fp.fit(img)
fp.plot()


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
img = fp.import_example()

# 2dpeaks example
fp = findpeaks()
fp.fit(img)
fp.plot()

# 2dpeaks example with other settings
fp = findpeaks(mask=0, scale=True, denoise=10, togray=True, resize=(300,300), verbose=3)
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
fp.fit(X)
fp.plot()

fp = findpeaks(lookahead=1, smooth=10, verbose=3)
fp.fit(X)
fp.plot()

# %%
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(lookahead=1, smooth=10)
fp.fit(X)
fp.plot()

# %%
from math import pi
import numpy as np

i = 10000
xs = np.linspace(0,3.7*pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Findpeaks
fp = findpeaks()
results=fp.fit(X)
fp.plot()

results['max_peaks_s']
results['min_peaks_s']
results['labx_s']
