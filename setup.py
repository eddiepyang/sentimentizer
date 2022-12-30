from setuptools import setup, find_packages

setup(
    packages=find_packages(where=".", exclude=["*.zip", "*.pyc"]),
    include_package_data=True,
)
