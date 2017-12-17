# -*- coding: utf-8 -*-
"""building saxpy."""

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="saxpy",
    version='0.1.0',
    url="https://github.com/seninp/saxpy.git",
    packages=['saxpy'],

    author="Pavel Senin",
    author_email="senin@hawaii.edu",

    description="SAX implementation for Python",
    long_description=open('README.rst').read(),

    install_requires=requirements,

    keywords='saxpy',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],

    license=license
)
