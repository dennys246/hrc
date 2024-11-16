from setuptools import setup, find_packages

setup(
    name='hrc',
    version='0.1.0',
    description='A Python library for convolving mne NIRS objects with a dynamically built hemodynamic response function to remove temporal blur from brain signals',
    author='Denny Schaedig',
    author_email='denny.schaedig@gmail.com',
    url='https://github.com/dennys246/hrc',
    packages=find_packages(),
    install_requires=[
        'requests',  # Example dependency
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)