# %%
# import os
# os.chdir(os.path.dirname(os.path.abspath('examples.py')))
# import findpeaks
# print(dir(findpeaks))
# print(findpeaks.__version__)

# pip install opencv-python
import matplotlib.pyplot as plt
# from findpeaks import findpeaks

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


# %%
import numpy as np
from scipy.ndimage import gaussian_filter
from findpeaks import findpeaks
rng = np.random.default_rng(42)
x = rng.normal(size=(50, 50))
x = gaussian_filter(x, sigma=10.)
# peak and valley
fp = findpeaks(method="topology", whitelist=['peak', 'valley'], denoise=None, verbose=3)
results = fp.fit(x)

fp.plot(figsize=(25, 15), figure_order='horizontal', cmap=plt.cm.hot_r)
fp.plot_persistence()
# fp.plot_mesh()

# %%

from findpeaks import findpeaks
path = r'https://user-images.githubusercontent.com/12035402/274193739-cdfd8986-91eb-4211-bef6-ebad041f47ae.png'
fp = findpeaks(method='topology', whitelist='peak', limit=5, denoise='lee_sigma', params={'window': 5})
X = fp.imread(path)
results = fp.fit(X)

result_df = results['persistence']
peak = result_df.index[result_df['peak']==True].tolist()
print(result_df.loc[peak])
print(result_df.shape)
fp.plot_persistence()
fp.plot(figsize=(25, 14), text=False, marker='x', color='#ff0000', figure_order='vertical')
# fp.plot_mesh(cmap=plt.cm.hot, view=(40, 180))
# fp.plot_mesh(view=(90, 0))

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

# %% Issue



# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=1, params={'delta': 1}, verbose=3)
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
path = r'https://user-images.githubusercontent.com/44827483/221152897-133839bb-7364-492a-921b-c9077ab9930b.png'
fp = findpeaks(method='topology', whitelist='peak', denoise='lee_enhanced', params={'window': 5})
fp = findpeaks(method='topology', whitelist='peak', denoise='lee_sigma', params={'window': 5})
X = fp.imread(path)
results = fp.fit(X)
fp.plot_persistence()
fp.plot()
# fp.plot_mesh()

fp.results['persistence'].iloc[0:10,:]

# %%
from findpeaks import findpeaks
X = [1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1]

fp = findpeaks(method='peakdetect', whitelist=['peak', 'valley'], lookahead=1, verbose=3)
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


# %% issue in mail
import numpy as np
from findpeaks import findpeaks

# Initialize peakdetect
fp1 = findpeaks(method='peakdetect', whitelist='peak', lookahead=200)

# Initialize topology
fp2 = findpeaks(method='topology')

# Example 1d-vector
i = 10000
xs = np.linspace(0,3.7*np.pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Fit using peakdetect
results_1 = fp1.fit(X)

# Fit using topology
results_2 = fp2.fit(X)

# Plot peakdetect
fp1.plot()

# Plot topology
fp2.plot()
fp2.plot_persistence(fontsize_ax1=None)


# %% issue #12
# https://github.com/erdogant/findpeaks/issues/12
import numpy as np

X = np.sin(np.linspace(0, 1, 100))
from findpeaks import findpeaks
# fp = findpeaks(method='caerus', params_caerus={'minperc': 5, 'window': 50})
fp = findpeaks(method='caerus', params={'minperc': 5, 'window': 50})
results = fp.fit(X)


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

# %% find peak and valleys in 2d images.
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from findpeaks import findpeaks

rng = np.random.default_rng(42)
x = rng.normal(size=(50, 50))
x = gaussian_filter(x, sigma=10)

fp = findpeaks(method="topology", whitelist=['peak', 'valley'], denoise=None, verbose=3)
results = fp.fit(x)
# Make plots
results['persistence']
fp.plot(cmap='coolwarm', text=True)

fp.plot_persistence()
fp.plot_mesh()

# %%
from findpeaks import findpeaks
import cv2
x = cv2.imread('C://temp/LbK2I.png')
# x = cv2.imread('C://temp/uISO2.png')
# x = [7193,7539,14310,14364,7414,7353,7557,7184,7188,7550,14330,14414,7480,7400,7582,7205,7140,7546,14356,14462,7533,7426,7618,7224,7192,7550,14371,14494,7558,7436,7634,7231,7172,7556,14390,14545,7616,7491,7669,7260,7123,7557,14404,14590,7671,7532,7698,7281,7172,7578,14396,14624,7703,7555,7718,7283,7214,7597,14407,14659,7729,7574,7742,7280,7219,7599,14414,14702,7764,7594,7743,7284,7217,7604,14414,14729,7794,7604,7740,7283,7233,7655,14430,14782,7853,7647,7779,7306,7207,7663,14444,14831,7895,7676,7808,7326,7292,7673,14427,14848,7889,7653,7769,7313,7330,7711,14433,14883,7909,7654,7772,7322,7343,7732,14442,14934,7969,7706,7807,7341,7315,7735,14450,14987,8043,7771,7837,7360,7330,7759,14458,15032,8083,7791,7857,7386,7364,7790,14464,15063,8106,7804,7859,7397,7364,7799,14474,15097,8161,7838,7873,7413,7374,7803,14468,15139,8212,7858,7891,7432,7391,7836,14478,15181,8258,7900,7927,7462,7405,7840,14459,15195,8278,7932,7950,7476,7389,7849,14454,15224,8321,7965,7978,7494,7470,7860,14445,15230,8322,7972,7979,7497,7430,7895,14454,15267,8344,8006,8020,7518,7420,7898,14440,15276,8332,7996,8015,7516,7482,7918,14410,15265,8332,7988,8005,7519,7536,7956,14404,15279,8339,7998,8026,7539,7523,7949,14410,15289,8329,8005,8047,7564,7501,7945,14403,15283,8329,8009,8049,7562,7533,7970,14392,15283,8321,8017,8068,7577,7515,7980,14409,15297,8319,8031,8078,7600,7508,7967,14396,15286,8331,8047,8072,7590,7582,7967,14362,15257,8336,8041,8072,7583,7650,7956,14344,15229,8318,8015,8060,7580,7570,7945,14342,15223,8314,8015,8059,7599,7529,7966,14355,15229,8321,8041,8085,7628,7578,7959,14352,15212,8322,8048,8089,7632,7546,7932,14332,15181,8306,8055,8078,7615,7494,7943,14325,15160,8291,8057,8069,7616,7547,7962,14296,15126,8253,8031,8053,7606,7646,7953,14239,15066,8203,7985,8015,7570,7539,7955,14249,15057,8196,7989,8031,7602,7447,7967,14276,15059,8191,8024,8069,7646,7532,7942,14258,14998,8145,7994,8046,7633,7543,7932,14220,14927,8073,7949,8011,7594,7491,7943,14190,14869,8019,7919,7992,7583,7505,7897,14183,14822,7985,7907,8007,7593,7470,7858,14176,14778,7940,7889,7991,7584,7478,7823,14152,14706,7876,7857,7960,7567,7549,7832,14111,14618,7783,7795,7915,7526,7456,7792,14132,14556,7723,7756,7910,7515,7336,7745,14149,14517,7669,7719,7881,7503,7309,7725,14143,14448,7587,7676,7840,7476,7350,7701,14117,14364,7496,7624,7798,7432,7295,7653,14114,14298,7440,7572,7746,7394,7282,7643,14100,14236,7350,7527,7704,7368,7296,7626,14081,14174,7284,7491,7688,7345,7217,7588,14086,14110,7205,7440,7655,7310,7174,7557,14086,14038,7139,7387,7613,7283,7209,7512,14081,13995,7087,7351,7578,7274,7178,7472,14078,13950,7029,7318,7549,7255,7110,7441,14066,13878,6954,7259,7489,7222,7076,7420,14076,13839,6896,7223,7485,7225,7127,7390,14050,13784,6831,7186,7469,7187,7013,7332,14083,13760,6783,7164,7458,7183,6996,7308,14078,13712,6725,7128,7417,7170,7024,7302,14077,13676,6668,7081,7392,7135,6996,7283,14071,13635,6613,7034,7363,7121,6940,7247,14067,13593,6553,6998,7349,7102,6892,7258,14098,13599,6539,7006,7384,7119,6894,7222,14086,13558,6483,6972,7346,7096,6920,7193,14064,13510,6437,6926,7301,7050,6937,7191,14053,13506,6422,6914,7295,7047,6928,7176,14060,13499,6393,6894,7292,7056,6855,7155,14077,13482,6384,6899,7281,7061,6861,7159,14069,13479,6372,6881,7266,7057,6935,7164,14063,13461,6334,6857,7253,7047,6935,7140,14047,13443,6303,6828,7218,7027,6897,7114,14054,13439,6296,6822,7200,7030,6801,7123,14090,13475,6316,6861,7245,7069,6763,7123,14087,13484,6316,6870,7268,7087,6961,7130,14023,13420,6250,6812,7202,7032,7013,7148,14022,13406,6231,6804,7201,7025,6947,7131,14057,13437,6247,6835,7236,7053,6806,7094,14081,13461,6263,6848,7262,7070,6828,7101,14059,13453,6250,6822,7240,7070,6870,7118,14054,13451,6256,6817,7249,7073,6882,7106,14045,13464,6253,6827,7237,7063,6925,7101,14028,13443,6243,6817,7216,7047,6954,7111,14036,13458,6254,6831,7224,7050,6924,7091,14038,13461,6253,6837,7221,7054,6868,7082,14049,13474,6257,6828,7214,7051,6866,7107,14052,13495,6271,6838,7231,7069,6866,7121,14046,13510,6284,6858,7243,7086,6868,7107,14023,13503,6260,6851,7216,7070,6976,7097,13991,13465,6223,6817,7162,7037,6980,7117,13999,13479,6244,6822,7191,7044,6827,7093,14059,13534,6302,6867,7234,7099,6831,7094,14064,13540,6302,6854,7217,7102,6876,7114,14052,13551,6290,6858,7231,7108]

fp = findpeaks(method="peakdetect", whitelist=['peak', 'valley'], limit=0, denoise=None, verbose=3)
results = fp.fit(x)

# results['df']
fp.plot()
fp.plot_persistence()
fp.plot_mesh(view=(90, 0))


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from findpeaks import findpeaks
rng = np.random.default_rng(42)

x = rng.normal(size=(50, 50))
x = gaussian_filter(x, sigma=10.)

# Search for peaks/valleys with a minimum value of 0
fp = findpeaks(method="topology", limit=0, denoise=None, verbose=3)
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
verbose=3

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

fp = findpeaks(method="caerus", interpolate=None, params={'minperc': 100}, verbose=verbose)
# Make fit
results = fp.fit(X)
ax = fp.plot()


# %%
from findpeaks import findpeaks
import numpy as np

# np.random.seed(100)
np.random.seed(200)
X = np.random.randint(200, size=400)

fp = findpeaks(method='topology', lookahead=1, interpolate=10, verbose=3)
results = fp.fit(X)

fig=fp.plot()
fp.plot_mesh()

# %%
from findpeaks import findpeaks
fp = findpeaks(method="topology", verbose=3, limit=1)
X = fp.import_example("2dpeaks")
results = fp.fit(X)
fp.plot(figure_order='horizontal')

# %%
# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# Initialize
fp = findpeaks(method='peakdetect', lookahead=1, verbose=3)
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
               params={'window': 31}, verbose=3)
# Fit
fp.fit(img)
fp.plot()
fp.results["persistence"]

# Take the minimum score for the top peaks off the diagonal.
limit = fp.results['persistence'][0:5]['score'].min()
fp.plot(text=True, limit=limit)

fp = findpeaks(whitelist=['peak', 'valley'], limit=limit, imsize=(300, 300), scale=True, togray=True, denoise='fastnl',
               params={'window': 31}, verbose=3)
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

# %%
from findpeaks import findpeaks
X = [10,11,9,23,21,11,45,20,11,12]
fp = findpeaks(method="topology", lookahead=1, verbose=3)
results = fp.fit(X)
fp.plot()
fp.plot_persistence()


# %% Run over all methods and many parameters
from findpeaks import findpeaks
savepath=''
methods = ['mask','topology', None]
filters = ['fastnl','bilateral','frost','median','mean', None]
windows = [3, 9, 15, 31, 63]
cus = [0.25, 0.5, 0.75]
verbose=3

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

fp = findpeaks(method='peakdetect', lookahead=1, interpolate=10, verbose=3)
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()
fp.plot_persistence()


from findpeaks import findpeaks
fp = findpeaks(method='topology', verbose=3)
X = fp.import_example('1dpeaks')
fp.fit(X)
fp.plot()
fp.plot_persistence()

from findpeaks import findpeaks
fp = findpeaks(method='topology', interpolate=10, verbose=3)
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

fp = findpeaks(method='mask', verbose=3)
img = fp.import_example()
fp.fit(img)
fp.plot()


# 2dpeaks example with other settings
fp = findpeaks(method='topology', imsize=(300, 300), scale=True, togray=True, denoise='fastnl', params={'window': 31},
               verbose=3)
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
fp = findpeaks(method='topology', imsize=False, scale=False, togray=True, denoise='fastnl', verbose=3)
fp.fit(img)
fp.plot_persistence()
fp.plot_mesh(wireframe=False, title='image_lee_enhanced', view=(30,30))

# %%
