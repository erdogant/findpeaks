
# %%
import matplotlib.pyplot as plt
import findpeaks
print(dir(findpeaks))
print(findpeaks.__version__)


# %%
X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
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
