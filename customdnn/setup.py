#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:04:59 2020

@author: manpreetsi
"""

from setuptools import setup

setup(name="customdnn",
      version = "0.1",
      description ="A custom deep neural network package that gives the liberty to design a customized neural network (using NumPy only)",
      url="https://github.com/singhmnprt01/Custom-Deep-Neural-Network",
      author="Manpreet Singh",
      packages = ['customdnn'],
      install_requires =['pandas','numpy','matplotlib','sklearn','datetime','sys'],
      long_description = "README"
      classifiers=[  # Optional
         'Development Status :: 3 - Alpha',

   
        'Intended Audience :: Deep Learning Learners',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: GNU', 
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
  ]     
      )


