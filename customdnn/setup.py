
"""
@author: singhmnprt01@gmail.com

customdnn Copyright (C) 2020 singhmnprt01@gmail.com
"""
from setuptools import setup

setup(name="customdnn",
      version = "0.4",
      description ="A custom deep neural network package that gives the liberty to design a customized neural network (using NumPy only)",
      url="https://github.com/singhmnprt01/Custom-Deep-Neural-Network",
      author="singhmnprt01@gmail.com",
      packages = ['customdnn'],
      install_requires =['pandas','numpy','matplotlib','sklearn','datetime'],
      long_description = "README",
      classifiers=[  
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Education',
         'Topic :: Software Development :: Build Tools',

         'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 
            
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8'
  ]     
      )


