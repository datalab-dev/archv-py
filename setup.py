from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='archv',
    version='0.1.0',
    packages=['archv', 'scripts'],
    long_description=open('README.txt').read(),
    install_requires=requirements,
)
