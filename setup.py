from setuptools import setup, find_packages

setup(
    name="trademodels",
    version="0.1.0",
    author="Pieter Pel",
    author_email="contact@pieterpel.com",
    description="Package used for quantitative trading modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PieterPel/trademodels",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "dill==0.3.8",
        "gymnasium==0.29.1",
        "matplotlib==3.8.2",
        "numpy==1.23.5",
        "pandas==2.2.1",
        "sb3-contrib==2.2.1",
        "scikit-learn==1.4.0",
        "stable-baselines3==2.2.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11.2",
    ],
    python_requires=">=3",
    license="BSD-3-Clause",
)
