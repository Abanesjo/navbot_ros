from setuptools import find_packages
from setuptools import setup

setup(
    name='navbot_ros',
    version='0.0.0',
    packages=find_packages(
        include=('navbot_ros', 'navbot_ros.*')),
)
