from setuptools import setup, find_packages

setup(name='discrete-autoencoder',
      version='0.0.1',
      install_requires=['Keras','theano'],
      author='Ben Striner',
      url='https://github.com/bstriner/discrete_autoencoder',
      packages=find_packages())
