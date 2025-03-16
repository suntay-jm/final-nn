from setuptools import setup, find_packages

setup(
    name="DnnA",
    version="0.1",
    packages=find_packages(),  # automatically finds all packages
    install_requires=[
        "numpy",  
        "pytest",
    ],
    python_requires=">=3.6",  
)