from setuptools import setup, find_packages

setup(
    name="numcompute-stream",
    version="0.2.0",
    description="A NumPy-based toolkit for numerical computation and streaming machine learning",
    packages=find_packages(include=["numcompute_stream", "numcompute_stream.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.9",
)
