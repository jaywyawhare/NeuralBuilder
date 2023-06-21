from setuptools import setup, find_packages

setup(
    name='neuralbuilder',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.12.0',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'scikit-learn>=1.2.2'
    ],
    author='jaywyawhare',
    description='A library for building deep learning models',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
