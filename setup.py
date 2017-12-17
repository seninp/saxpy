# -*- coding: utf-8 -*-
"""building saxpy."""

from setuptools import setup


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="saxpy",
    version='0.1.0',
    url="https://github.com/seninp/saxpy.git",
    packages=['saxpy'],

    description="SAX implementation for Python",
    long_description=open('README.rst').read(),
    keywords='saxpy',

    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    author="Pavel Senin",
    author_email="senin@hawaii.edu",

    license=license
)
