# %% TODO 2D-peaks
# https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
# https://www.sthu.org/code/codesnippets/imagepers.html
# https://stackoverflow.com/questions/46000842/detecting-peaks-in-images/47191052#47191052

# %%
import findpeaks
print(dir(findpeaks))
print(findpeaks.__version__)

# %%
import cv2
import matplotlib.pyplot as plt
f = 'D://GITLAB/DATA/tmp/0421_2012_001_cropped.png'
# f = 'D://GITLAB/DATA/tmp/0411_2012_001_cropped_met_rood.png'
img = cv2.imread(f)
# img = cv2.resize(img, (200, 200))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.figure()
# plt.imshow(img)
# img = img.astype(np.uint8)
# img = img / 255
# plt.imshow(img, cmap='gray')
# img = cv2.blur(img,(3,3))
# plt.figure()
# plt.imshow(img)

# img = cv2.fastNlMeansDenoisingColored(img, h=20)
# plt.figure()
# plt.imshow(img)

# convert all to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.fastNlMeansDenoising(img, h=20)
# plt.figure()
# plt.imshow(img, cmap='gray')

# %%
results = findpeaks.fit(img, mask=0, scale=True, denoise=30, togray=True, resize=(300,300), verbose=3)
# findpeaks.plot(results)
findpeaks.plot_preprocessing(results)
findpeaks.plot_mask(results)
findpeaks.plot_mesh(results)
findpeaks.plot_peristence(results)

# %%
df = findpeaks.import_example()
results = findpeaks.fit(df.values, mask=0, scale=True, togray=True, denoise=None, verbose=3)
findpeaks.plot(results)

# findpeaks.plot_preprocessing(results)
# findpeaks.plot_mask(results)
# findpeaks.plot_mesh(results)
# findpeaks.plot_peristence(results)

# %%
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]
out = findpeaks.fit(X, lookahead=1)
findpeaks.plot(out)

out = findpeaks.fit(X, lookahead=1, smooth=10)
findpeaks.plot(out)



# %%
X = [10,11,9,23,21,11,45,20,11,12]
out = findpeaks.fit(X, lookahead=1, smooth=10)
findpeaks.plot(out)

# %%
from math import pi
import numpy as np

i = 10000
xs = np.linspace(0,3.7*pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * 
np.random.randn(i))
y *= -1

# Findpeaks
out = findpeaks.fit(X)
findpeaks.plot(out)

# Or alternatively
_max, _min = peakdetect(X, xs)
xm = [p[0] for p in _max]
ym = [p[1] for p in _max]
xn = [p[0] for p in _min]
yn = [p[1] for p in _min]

plt.plot(x, y)
plt.plot(xm, ym, "r+")
plt.plot(xn, yn, "g+")
