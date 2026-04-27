"""ScalarGrad - Automatic Differentiation Engine Setup"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scalargrad",
    version="0.1.0",
    author="Adarsh Tiwari",
    author_email="adarsh@example.com",
    description="A minimal scalar-based automatic differentiation engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adarshLabs/ScalarGrad",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    keywords="autodiff autograd backpropagation neural networks machine learning",
    project_urls={
        "Bug Reports": "https://github.com/adarshLabs/ScalarGrad/issues",
        "Source": "https://github.com/adarshLabs/ScalarGrad",
        "Documentation": "https://github.com/adarshLabs/ScalarGrad#readme",
    },
)
