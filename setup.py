import setuptools
import re

# versioning ------------
VERSIONFILE="findpeaks/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Setup ------------
with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['scipy',
                       'matplotlib',
                       'numpy',
                       'pandas',
                       'tqdm',
                       'requests',
                       'caerus>=0.1.9',
                       'xarray',
                       'joblib'],
     dependency_links=['https://github.com/arvinnick/peakdetect/tarball/master#egg=peakdetect-1.2'],
     python_requires='>=3',
     name='findpeaks',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="findpeaks is for the detection of peaks and valleys in a 1D vector and 2D array (image).",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/findpeaks",
	 download_url = 'https://github.com/erdogant/findpeaks/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
