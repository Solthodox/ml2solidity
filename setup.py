from setuptools import setup, find_packages

setup(
    name="ml2solidity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
    ],
    author="Solthodox",
    author_email="solthodox@gmail.com",
    description="Export scikit-learn Random Forest models to Solidity smart contracts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/solthodox/ml2solidity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)