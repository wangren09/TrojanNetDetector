
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


setup(
  name='robustness',

  version='9',

  description='Tools for Robustness',


  classifiers=[

        'Development Status :: 4 - Beta',


        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',


        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
  ],



  packages=find_packages(),
  include_package_data=True,
  package_data={
            'certificate': ['client/server.crt']
  },


  install_requires=['tqdm', 'grpcio', 'psutil', 'gitpython','py3nvml'],
)
