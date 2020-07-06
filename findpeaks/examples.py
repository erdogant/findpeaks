# %%
import findpeaks
print(dir(findpeaks))
print(findpeaks.__version__)

# %%
import cv2
f = 'D://GITLAB/DATA/tmp/0421_2012_001_cropped.png'
# f = 'D://GITLAB/DATA/tmp/0411_2012_001_cropped_met_rood.png'
# f = 'D://GITLAB/DATA/tmp/6272_2012_002_cropped.png'
# f = 'D://GITLAB/DATA/tmp/6353_2012_001_cropped.png'

img = cv2.imread(f)

# %%
from findpeaks import findpeaks
fp = findpeaks(mask=0, scale=True, denoise=30, togray=True, resize=(300,300), verbose=3)
# fp = findpeaks(mask=0, scale=True, denoise=None, togray=True, resize=(300,300), verbose=3)
results = fp.fit(img)
fp.plot()

# Plot each seperately
fp.plot_preprocessing()
fp.plot_mask()
fp.plot_peristence()
fp.plot_mesh()

# Make mesh plot
fp.plot_mesh(view=(0,90))
fp.plot_mesh(view=(90,0))

# %%
import findpeaks
df = fp.import_example()
results = findpeaks.fit(df.values, mask=0, verbose=3)
results = findpeaks.fit(df.values, mask=0, scale=True, togray=False, denoise=3, verbose=3)
findpeaks.plot(results)

# findpeaks.plot_preprocessing(results)
# findpeaks.plot_mask(results)
# findpeaks.plot_mesh(results)
# findpeaks.plot_peristence(results)

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
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * 
np.random.randn(i))
# y *= -1

# Findpeaks
fp = findpeaks()
fp.fit(X)
fp.plot()
