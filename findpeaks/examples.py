
# %%
import matplotlib.pyplot as plt
import findpeaks as findpeaks
print(dir(findpeaks))
print(findpeaks.__version__)


# %%
X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
out = findpeaks.fit(X, smooth=10)
findpeaks.plot(out)


# %%
X = [10,11,9,23,21,11,45,20,11,12]
out = findpeaks.fit(X, smooth=10)
findpeaks.plot(out)

# %%
