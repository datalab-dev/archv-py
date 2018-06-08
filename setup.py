from setuptools import setup

setup(
    name='archv',
    version='0.1.0',
    packages=['archv', 'scripts'],
    long_description=open('README.txt').read(),
    install_requires=['pyyaml', 'opencv-contrib-python', 'opencv-python'],
)
