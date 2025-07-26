# findpeaks

[![Python](https://img.shields.io/pypi/pyversions/findpeaks)](https://img.shields.io/pypi/pyversions/findpeaks)
[![Pypi](https://img.shields.io/pypi/v/findpeaks)](https://pypi.org/project/findpeaks/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/findpeaks/)
[![LOC](https://sloc.xyz/github/erdogant/findpeaks/?category=code)](https://github.com/erdogant/findpeaks/)
[![Downloads](https://static.pepy.tech/personalized-badge/findpeaks?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/findpeaks)
[![Downloads](https://static.pepy.tech/personalized-badge/findpeaks?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/findpeaks)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/findpeaks/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/findpeaks.svg)](https://github.com/erdogant/findpeaks/network)
[![Issues](https://img.shields.io/github/issues/erdogant/findpeaks.svg)](https://github.com/erdogant/findpeaks/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/260400472.svg)](https://zenodo.org/badge/latestdoi/260400472)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/findpeaks/pages/html/Documentation.html#colab-notebook)
[![Medium](https://img.shields.io/badge/Medium-Blog-green)](https://erdogant.github.io/findpeaks/pages/html/Documentation.html#medium-blog)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/findpeaks/pages/html/Documentation.html#)

<div>
<a href="https://erdogant.github.io/findpeaks/"><img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/logo.png" width="125" align="left" /></a>
findpeaks is a comprehensive Python library for robust detection and analysis of peaks and valleys in both 1D vectors and 2D arrays (images). The library provides multiple detection algorithms including topology-based persistent homology (most robust), mask-based local maximum filtering, and traditional peakdetect approaches. It can be used for time series analysis, signal processing, image analysis, and spatial data. ⭐️Star it if you like it⭐️
</div>

---

### Key Features

| Feature | Description |
|--------|-------------|
| [**Topology Detection**](https://erdogant.github.io/findpeaks/pages/html/Topology.html) | Mathematically grounded peak detection using persistent homology. |
| [**Peakdetect Method**](https://erdogant.github.io/findpeaks/pages/html/Peakdetect.html) | Traditional peak detection algorithm for noisy signals. |
| [**Mask Detection**](https://erdogant.github.io/findpeaks/pages/html/Mask.html) | Local maximum filtering for 2D image analysis. |
| [**Caerus Method**](https://erdogant.github.io/findpeaks/pages/html/Caerus.html) | Specialized algorithm for financial time series analysis. |
| [**Preprocessing**](https://erdogant.github.io/findpeaks/pages/html/Pre-processing.html) | Denoising, scaling, interpolation, and image preprocessing. |
| [**Visualization**](https://erdogant.github.io/findpeaks/pages/html/Plots.html) | Rich plotting capabilities including persistence diagrams and 3D mesh plots. |

---

### Resources and Links
- **Example Notebooks:** [Examples](https://erdogant.github.io/findpeaks/pages/html/Examples.html)
- **Blog Posts:** [Medium](https://erdogant.medium.com)
- **Documentation:** [Website](https://erdogant.github.io/findpeaks)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/erdogant/findpeaks/issues)

---

### Background

* **Topology Method**: The most robust detection method based on persistent homology from topological data analysis. It quantifies peak significance through persistence scores and provides mathematically stable results even in noisy data.

* **Peakdetect Method**: Traditional algorithm that excels at finding local maxima and minima in noisy signals without requiring extensive preprocessing. Uses a lookahead approach to distinguish between true peaks and noise-induced fluctuations.

* **Mask Method**: Local maximum filtering approach specifically designed for 2D data (images). Employs 8-connected neighborhood analysis and background removal for spatial peak detection.

* **Preprocessing Pipeline**: Comprehensive preprocessing capabilities including interpolation, denoising (Lee, Frost, Kuan filters), scaling, and image resizing to improve detection accuracy.

---

### Installation

##### Install findpeaks from PyPI
```bash
pip install findpeaks
```

##### Install from Github source
```bash
pip install git+https://github.com/erdogant/findpeaks
```  

##### Import Library
```python
import findpeaks
print(findpeaks.__version__)

# Import library
from findpeaks import findpeaks
```

---

### Quick Start

```python
# Import library
from findpeaks import findpeaks

# Initialize with topology method (most robust)
fp = findpeaks(method='topology')

# Example data
X = fp.import_example('1dpeaks')

# Peak detection
results = fp.fit(X)

# Plot results
fp.plot()

# Plot persistence diagram
fp.plot_persistence()
```

---

### Examples

#### 1D Signal Analysis
* [Find peaks in low sampled dataset](https://erdogant.github.io/findpeaks/pages/html/Examples.html#find-peaks-in-low-sampled-dataset)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Examples.html#find-peaks-in-low-sampled-dataset">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_raw.png" width="400" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_interpol.png" width="400" />  
  </a>
</p>

* [Comparison of peak detection methods](https://erdogant.github.io/findpeaks/pages/html/Examples.html#comparison-methods-1)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Examples.html#comparison-methods-1">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_peakdetect_int.png" width="400" />  
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_topology_int.png" width="400" />    
  </a>
</p>

* [Find peaks in high sampled dataset](https://erdogant.github.io/findpeaks/pages/html/Examples.html#find-peaks-in-high-sampled-dataset)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Examples.html#find-peaks-in-high-sampled-dataset">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig3.png" width="600" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Examples.html#find-peaks-in-high-sampled-dataset">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig3_persistence_limit.png" width="600" />
  </a>
</p>

#### 2D Image Analysis
* [Find peaks in an image (2D-array)](https://erdogant.github.io/findpeaks/pages/html/Examples.html#d-array-image)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Examples.html#d-array-image">
 <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_raw.png" width="115" />
 <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mask.png" width="500" />
  </a>
</p>

* [3D mesh visualization](https://erdogant.github.io/findpeaks/pages/html/Plots.html#d-mesh)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Plots.html#d-mesh">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mesh1.png" width="400" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/2dpeaks_mesh2.png" width="400" />
  </a>
</p>

#### Financial Time Series
* [Bitcoin price analysis](https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#bitcoin)
* [Facebook stock analysis](https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#facebook-stocks)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#facebook-stocks">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig_facebook_minperc5.png" width="600" />
  </a>
</p>

#### SAR/SONAR Image Processing
* [Peak detection in SAR/SONAR images](https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#sonar)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#sonar">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/sonar_plot.png" width="600" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#sonar">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/sonar_mesh1.png" width="300" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/sonar_mesh2.png" width="300" />
  </a>
</p>

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Use-cases.html#sonar">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/sonar_mesh3.png" width="300" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/sonar_mesh4.png" width="300" />
  </a>
</p>

#### Image Denoising
* [Denoising with Lee, Kuan, Fastnl, Bilateral, Frost, Mean, Median filters](https://erdogant.github.io/findpeaks/pages/html/Denoise.html#)

<p align="left">
  <a href="https://erdogant.github.io/findpeaks/pages/html/Denoise.html#">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/noise_distr_examples.png" width="600" />
  </a>
</p>

<hr>

### References
* https://github.com/erdogant/findpeaks
* https://github.com/Anaxilaus/peakdetect
* https://www.sthu.org/blog/13-perstopology-peakdetection/index.html

### Contributors
Special thanks to the contributors!

<p align="left">
  <a href="https://github.com/erdogant/findpeaks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erdogant/findpeaks" />
  </a>
</p>

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free** but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
