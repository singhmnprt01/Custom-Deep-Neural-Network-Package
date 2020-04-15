
"""
@author: singhmnprt01
"""

from setuptools import setup

setup(name="customdnn",
      version = "0.1",
      description ="A custom deep neural network package that gives the liberty to design a customized neural network (using NumPy only)",
      url="https://github.com/singhmnprt01/Custom-Deep-Neural-Network",
      author="singhmnprt01",
      packages = ['customdnn'],
      install_requires =['pandas','numpy','matplotlib','sklearn','datetime','sys'],
      long_description = "README"
      classifiers=[  
         'Development Status :: 3 - Alpha',

        'Intended Audience :: Deep Learning Learners',
        'Topic :: Software Development :: Build Tools',

        'License :: GNU', 
            
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
  ]     
      )


