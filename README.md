# findpeaks

[![Python](https://img.shields.io/pypi/pyversions/findpeaks)](https://img.shields.io/pypi/pyversions/findpeaks)
[![PyPI Version](https://img.shields.io/pypi/v/findpeaks)](https://pypi.org/project/findpeaks/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/findpeaks/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/findpeaks/month)](https://pepy.tech/project/findpeaks/month)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/findpeaks/)

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
import findpeaks as findpeaks
```

#### Example 1:
```python
X = [9,60,377,985,1153,672,501,1068,1110,574,135,23,3,47,252,812,1182,741,263,33]
out = findpeaks.fit(X)
findpeaks.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_raw.png" width="600" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig1_interpol.png" width="600" />  
</p>

#### Example 2:
```python
X = [10,11,9,23,21,11,45,20,11,12]
out = findpeaks.fit(X)
findpeaks.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_raw.png" width="600" />
  <img src="https://github.com/erdogant/findpeaks/blob/master/docs/figs/fig2_interpol.png" width="600" />  
</p>


#### Citation
Please cite findpeaks in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020findpeaks,
  title={findpeaks},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/findpeaks}},
}
```

#### References
* https://github.com/erdogant/findpeaks

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
	See [LICENSE](LICENSE) for details.
	This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
