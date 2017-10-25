import setuptools

setuptools.setup(
    name="saxpy",
    version="0.1.0",
    url="https://github.com/seninp/saxpy.git",

    author="Pavel Senin",
    author_email="senin@hawaii.edu",

    description="SAX implementation for Python",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
