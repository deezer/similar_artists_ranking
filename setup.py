from setuptools import setup
from setuptools import find_packages

setup(name='src',
      description='similar_artists_ranking',
      author='Deezer Research',
      install_requires=['networkx==2.2',
                        'numpy',
                        'scikit-learn==0.22.1',
                        'scipy'],
      package_data={'src': ['README.md']},
      packages=find_packages())