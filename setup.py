from setuptools import setup, find_packages

setup(
    name="numcompute",
    version="0.1.0",
    description="A NumPy-based toolkit for numerical computation",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.9",
)