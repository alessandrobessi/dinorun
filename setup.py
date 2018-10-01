# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dinorun',
    version='0.1.0',
    description='Dinorun Reinforcement Learning',
    long_description=readme,
    author='Alessandro Bessi',
    author_email='alessandro.bessi@mail.com',
    url='https://github.com/alessandrobessi/dinorun',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
