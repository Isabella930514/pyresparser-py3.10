from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name='pyresparser-py3.10',
    version='2.0.0',
    url='https://github.com/sinmentis/pyresparser-py3.10',
    author='sinmentis',
    license='GPL-3.0',
    include_package_data=True,
    packages=find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    entry_points={
        'console_scripts': ['pyresparser=pyresparser.command_line:main'],
    }
)
