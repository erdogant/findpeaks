# =============================================================================
# PART 1
# =============================================================================
import matplotlib.pyplot as plt
# Import library
from findpeaks import findpeaks
# Initialize
fp = findpeaks(method='topology')
# Example 1d-vector
X = fp.import_example('1dpeaks')

# Plot
plt.figure(figsize=(15, 8), dpi=100)
plt.plot(X)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Value')

# Fit topology method on the 1D vector
results = fp.fit(X)

# Plot the results
fp.plot_persistence(figsize=(25,8))

# %%
# =============================================================================
# TOPOLOGY PART 2
# =============================================================================

# Import findpeaks
from findpeaks import findpeaks

# URL to image
path = r'https://erdogant.github.io/datasets/images/complex_peaks.png'
# Read image from url
X = fp.imread(path)

# Set findpeaks with its parameters
fp = findpeaks(method='topology', whitelist='peak', limit=5, denoise='lee_sigma', params={'window': 5})

# Detect peaks
results = fp.fit(X)

# Show persistence plot
fp.plot_persistence()
# Show plot
fp.plot(figsize=(25, 14), text=True, marker='x', color='#ff0000', figure_order='vertical', fontsize=14)# Mesh plot
fp.plot_mesh(view=(40, 225), cmap='hot')

# Results in dataframe
result_df = results['persistence']

"""
      x    y  birth_level  death_level       score  peak valley
0   433   95       255.00     0.000000  255.000000  True  False
1   310   95       255.00     8.920000  246.080000  True  False
2    62   95       255.00     8.960000  246.040000  True  False
3   186   95       255.00     9.680000  245.320000  True  False
4   457  100        85.00    33.600000   51.400000  True  False
5    39  100        85.00    33.640000   51.360000  True  False
6   163  100        85.00    34.040000   50.960000  True  False
7   334  100        85.00    34.160000   50.840000  True  False
8   210  100        85.00    34.240000   50.760000  True  False
9   286  100        85.00    34.320000   50.680000  True  False
...
...
46   13   50        15.16     9.833333    5.326667  True  False
"""

# %%
# =============================================================================
# MARK 2D
# =============================================================================

# Import library
from findpeaks import findpeaks
import matplotlib.pyplot as plt

# Initialize
fp = findpeaks(method='mask')
# Example 2d image
X = fp.import_example('2dpeaks')
# Plot RAW input image
plt.imshow(X)
# Fit using mask method
results = fp.fit(X)
# Plot the pre-processing steps
fp.plot_preprocessing()

# The output contains multiple variables
print(results.keys())
# dict_keys(['Xraw', 'Xproc', 'Xdetect'])

# Plot detected peaks
fp.plot(figure_order='horizontal')

# Create mesh plot from 2D image.
fp.plot_mesh()

# Rotate to make a top view
fp.plot_mesh(view=(90,0))


# %%
# =============================================================================
# PEAKDETECT
# =============================================================================
# Import libraries
import numpy as np
from findpeaks import findpeaks

# Create example data set
i = 10000
xs = np.linspace(0,3.7*np.pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))
# Initialize
fp = findpeaks(method='peakdetect', lookahead=200, interpolate=None)
# Fit peakdetect method
results = fp.fit(X)
# Plot
fp.plot()

# %%

# =============================================================================
# CERAUS
# =============================================================================

# Import library
from findpeaks import findpeaks

# Initialize findpeaks with cearus method.
# The default setting is that it only return peaks-vallyes with at least 5% difference.
fp = findpeaks(method='caerus', params={'minperc':5, 'window':50})
# Import example data
X = fp.import_example('facebook')
# Fit
results = fp.fit(X)
# Make the plot
fp.plot()

# %%

# =============================================================================
# SAR
# =============================================================================

# Import library
from findpeaks import findpeaks
# Initializatie
fp = findpeaks(scale=None, denoise=None, togray=True, imsize=(300, 300))
# Import image example
img = fp.import_example('2dpeaks_image')
# Fit
fp.fit(img)
# Tens of thousands of peaks are detected at this point. Better to put text=False
fp.plot(figure_order='horizontal', text=False)
fp.plot_mesh()

# %%

# =============================================================================
# TOPOLOGY
# =============================================================================
# Import library
from findpeaks import findpeaks
# Initializatie
fp = findpeaks(method='topology',
               togray=True,
               imsize=(300, 300),
               scale=True,
               denoise='fastnl',
               params={'window': 31})

# Import image example
img = fp.import_example('2dpeaks_image')
# Fit
fp.fit(img)
# Plot
fp.plot_preprocessing()

"""
[findpeaks] >Import [.\findpeaks\data\2dpeaks_image.png]
[findpeaks] >Finding peaks in 2d-array using topology method..
[findpeaks] >Resizing image to (300, 300).
[findpeaks] >Scaling image between [0-255] and to uint8
[findpeaks] >Conversion to gray image.
[findpeaks] >Denoising with [fastnl], window: [31].
[findpeaks] >Detect peaks using topology method with limit at None.
[findpeaks] >Fin.
"""

# Plot the top 15 peaks that are detected and examine the scores
fp.results['persistence'][1:5]

#      x    y  birth_level  death_level  score  peak  valley
# 2  131   52        166.0        103.0   63.0  True   False
# 3  132   61        223.0        167.0   56.0  True   False
# 4  129   60        217.0        194.0   23.0  True   False
# 5   40  288        113.0         92.0   21.0  True   False
# 6   45  200        104.0         87.0   17.0  True   False
# 7   87  293        112.0         97.0   15.0  True   False
# 8  165  110         93.0         78.0   15.0  True   False
# 9  140   45        121.0        107.0   14.0  True   False

# Take the minimum score for the top peaks off the diagonal.
limit = fp.results['persistence'][0:5]['score'].min()
# Plot
fp.plot(limit=limit, figure_order='horizontal')
# Mesh plot
fp.plot_mesh()
