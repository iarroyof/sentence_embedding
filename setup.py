from distutils.core import setup
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "wisse",
    description='Package for building sentence vector representations (sentence embeddings) from text.',
    version = "0.0.4",
    author = "Ignacio Arroyo-Fernandez",
    author_email='iaf@ciencias.unam.mx',
    url='https://github.com/iarroyof/sentence_embedding',
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=['wisse'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)


#setup(name='wisse',
#  version='1.0',
#  description='Package for building sentence vector representations (sentence embeddings) from text.',
#  author='Ignacio Arroyo-Fernandez',
#  author_email='iaf@ciencias.unam.mx',
#  url='https://github.com/iarroyof/sentence_embedding',
  #packages=['gensim', 'sklearn'],
#  py_modules = ['wisse', 'wisse.wisse'],
#  scripts=['wisse_example.py', 'keyed2indexed.py']
# )
