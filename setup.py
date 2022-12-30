from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["*.zip", "*.pyc"]),
    include_package_data=True,
)
