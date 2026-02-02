# -*- coding: utf-8 -*-
"""
Setup script for DAD-Net package.

Installation:
    pip install -e .

Or for development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="dad-net",
    version="1.0.0",
    author="Lan Zhang, Xianye Ben",
    author_email="",
    description="DAD-Net: Distribution-Aligned Dual-Stream Framework for Cross-Domain Micro-Expression Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/DAD-Net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "full": [
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "opencv-python>=4.5.0",
            "opencv-contrib-python>=4.5.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "adabelief-pytorch>=0.2.1",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dad-net-train=train:main",
            "dad-net-infer=inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
