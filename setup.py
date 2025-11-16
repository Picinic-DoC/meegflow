#!/usr/bin/env python3
"""Setup script for the NICE EEG Preprocessing Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="nice-preprocessing",
    version="0.1.0",
    description="A modular, configuration-driven EEG preprocessing pipeline using MNE-BIDS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NICE EEG Preprocessing Team",
    url="https://github.com/Laouen/nice-preprocessing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "eeg-preprocess=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
