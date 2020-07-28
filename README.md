# findpeaks

[![Python](https://img.shields.io/pypi/pyversions/findpeaks)](https://img.shields.io/pypi/pyversions/findpeaks)
[![PyPI Version](https://img.shields.io/pypi/v/findpeaks)](https://pypi.org/project/findpeaks/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/findpeaks/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/findpeaks)](https://pepy.tech/project/findpeaks)

* findpeaks is Python package

### Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install findpeaks from PyPI (recommended). findpeaks is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```python
conda create -n env_findpeaks python=3.7
conda activate env_findpeaks
```

```bash
pip install findpeaks
```

* Alternatively, install findpeaks from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/findpeaks.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/findpeaks#egg=master

# By cloning
pip install git+https://github.com/erdogant/findpeaks
git clone https://github.com/erdogant/findpeaks.git
cd findpeaks
python setup.py install
```  

#### Import findpeaks package
```python
import findpeaks
```

#### Example 1: 1D-vector low resolution

```python
# Load library
from findpeaks import findpeaks
# Data
X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
# Initialize
fp = findpeaks(lookahead=1)
results = fp.fit(X)
# Plot
fp.plot()
```

<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_raw.png" width="400" />
</p>

```python
# Initialize with interpolation parameter
fp = findpeaks(lookahead=1, interpolate=10)
results = fp.fit(X)
fp.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_interpol.png" width="400" />  
</p>

#### Example 2: 1D vector low resolution

```python
# Load library
from findpeaks import findpeaks
# Data
X = [10,11,9,23,21,11,45,20,11,12]
# Initialize
fp = findpeaks(lookahead=1)
results = fp.fit(X)
# Plot
fp.plot()
```

<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_raw.png" width="400" />
</p>

```python
# Initialize with interpolate parameter
fp = findpeaks(lookahead=1, interpolate=10)
results = fp.fit(X)
fp.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_interpol.png" width="400" />  
</p>


#### Example 3: 1D-vector high resolution

```python
# Load library
import numpy as np
from findpeaks import findpeaks

# Data
i = 10000
xs = np.linspace(0,3.7*np.pi,i)
X = (0.3*np.sin(xs) + np.sin(1.3 * xs) + 0.9 * np.sin(4.2 * xs) + 0.06 * np.random.randn(i))

# Initialize
fp = findpeaks()
results = fp.fit(X)

# Plot
fp.plot()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig3.png" width="600" />
</p>


#### Example 4: 2D-array (image) using default settings

```python
# Import library
from findpeaks import findpeaks

# Import example
X = fp.import_example()
print(X)
array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.4, 0.4],
       [0. , 0. , 0. , 0. , 0. , 0. , 0.7, 1.4, 2.2, 1.8],
       [0. , 0. , 0. , 0. , 0. , 1.1, 4. , 6.5, 4.3, 1.8],
       [0. , 0. , 0. , 0. , 0. , 1.4, 6.1, 7.2, 3.2, 0.7],
       [..., ..., ..., ..., ..., ..., ..., ..., ..., ...],
       [0. , 0.4, 2.9, 7.9, 5.4, 1.4, 0.7, 0.4, 1.1, 1.8],
       [0. , 0. , 1.8, 5.4, 3.2, 1.8, 4.3, 3.6, 2.9, 6.1],
       [0. , 0. , 0.4, 0.7, 0.7, 2.5, 9. , 7.9, 3.6, 7.9],
       [0. , 0. , 0. , 0. , 0. , 1.1, 4.7, 4. , 1.4, 2.9],
       [0. , 0. , 0. , 0. , 0. , 0.4, 0.7, 0.7, 0.4, 0.4]])

# Initialize
fp = findpeaks(mask=0)

# Fit
fp.fit(X)

# Plot the pre-processing steps
fp.plot_preprocessing()

# Plot all
fp.plot()

```

The input figure
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_raw.png" width="100" />
</p>

The masking approach detects the correct peaks.
```python
fp.plot_mask()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mask.png" width="600" />
</p>

Conversion from 2d to 3d mesh plots looks very nice. But there is a rough surface because of the low-resolution input data.
```python
fp.plot_mesh()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mesh1.png" width="600" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mesh2.png" width="600" />
</p>

The persistence plot appears to detect the right peaks.
```python
fp.plot_peristence()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_pers.png" width="600" />
</p>


#### Example 5: 2D-array (image) with pre-processing steps

```python
# Import library
from findpeaks import findpeaks

# Import example
X = fp.import_example()

# Initialize
fp = findpeaks(mask=0, scale=True, denoise=10, togray=True, size=(300,300), verbose=3)

# Fit
fp.fit(X)

# Plot all
fp.plot()

```

Show the plots:

```python
fp.plot_preprocessing()
```

<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_raw.png" width="100" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_raw_resized.png" width="100" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_raw_processed.png" width="100" />
</p>

The masking does not work so well because the pre-processing steps includes some weighted smoothing which is not ideal for the masking approach.
```python
fp.plot_mask()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mask_proc.png" width="600" />
</p>

The mesh plot has higher resolution because the pre-processing steps caused some smoothing.
```python
fp.plot_mesh()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_meshs1.png" width="600" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_meshs2.png" width="600" />
</p>

The Persistence plot does show the detection of correct peaks.
```python
fp.plot_peristence()
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_perss.png" width="600" />
</p>



#### Citation
Please cite findpeaks in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020findpeaks,
  title={findpeaks},
  author={Erdogan Taskesen},
  year={2020},
  howpublished={\url{https://github.com/erdogant/findpeaks}},
}
```

#### References
* https://github.com/erdogant/findpeaks
* https://github.com/Anaxilaus/peakdetect
* https://www.sthu.org/blog/13-perstopology-peakdetection/index.html

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
	See [LICENSE](LICENSE) for details.
