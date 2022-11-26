#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

test_requirements = [ ]

setup(
    author="Everinfer",
    author_email='hello@everinfer.ai',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Everinfer client to run scalable distributed ML inference with low latency.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='everinfer',
    name='everinfer',
    packages=find_packages(include=['everinfer', 'everinfer.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/everinfer/everinfer',
    version='0.1.5',
    zip_safe=False,
)
