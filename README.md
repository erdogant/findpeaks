# XXX

[![Python](https://img.shields.io/pypi/pyversions/XXX)](https://img.shields.io/pypi/pyversions/XXX)
[![PyPI Version](https://img.shields.io/pypi/v/XXX)](https://pypi.org/project/XXX/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/XXX/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/XXX/month)](https://pepy.tech/project/XXX/month)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/XXX/)

* XXX is Python package

### Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install XXX from PyPI (recommended). XXX is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment is created as following: 

```python
conda create -n env_XXX python=3.6
conda activate env_XXX
pip install -r requirements
```

```bash
pip install XXX
```

* Alternatively, install XXX from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/XXX.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/XXX#egg=master

# By cloning
pip install git+https://github.com/erdogant/XXX
git clone https://github.com/erdogant/XXX.git
cd XXX
python setup.py install
```  

#### Import XXX package
```python
import XXX as XXX
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/XXX/data/example_data.csv')
model = XXX.fit(df)
G = XXX.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/XXX/blob/master/docs/figs/fig1.png" width="600" />
  
</p>


#### Citation
Please cite XXX in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2020XXX,
  title={XXX},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/XXX}},
}
```

#### References
* 

### Maintainer
	Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
	Contributions are welcome.
	See [LICENSE](LICENSE) for details.
	This work is created and maintained in my free time. If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated.
