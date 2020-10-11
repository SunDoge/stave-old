import setuptools
from setuptools import version
from stave._version import __version__
from pathlib import Path

setuptools.setup(
    name='stave',
    version=__version__,
    description='deep learning framework for JAX',
    url='https://github.com/SunDoge/stave',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7',
    packages=setuptools.find_packages(exclude=['tests']),
)
