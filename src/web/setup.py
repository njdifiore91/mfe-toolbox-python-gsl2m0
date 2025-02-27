#!/usr/bin/env python3
"""
Setup script for the web/GUI component of the MFE Toolbox.
"""
import setuptools
from setuptools import find_packages
from pathlib import Path

# Package metadata
PACKAGE_NAME = "mfe-web"
VERSION = "4.0.0"
DESCRIPTION = "Web/GUI component of the MFE Toolbox providing interactive modeling environment"
AUTHOR = "Kevin Sheppard"
AUTHOR_EMAIL = "kevin.sheppard@economics.ox.ac.uk"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.12"

# File paths
README_PATH = Path("README.md")
REQUIREMENTS_PATH = Path("requirements.txt")
DEV_REQUIREMENTS_PATH = Path("requirements-dev.txt")


def read_requirements(requirements_file: Path) -> list:
    """
    Helper function to read package requirements from requirements files.

    Parameters
    ----------
    requirements_file : Path
        Path to the requirements file

    Returns
    -------
    list
        List of package requirements strings
    """
    if not requirements_file.exists():
        return []

    with open(requirements_file, "r") as f:
        requirements = f.readlines()

    # Filter out comments and empty lines
    requirements = [
        line.strip() for line in requirements
        if line.strip() and not line.strip().startswith("#")
    ]

    return requirements


def read_long_description() -> str:
    """
    Helper function to read long description from README.md.

    Returns
    -------
    str
        Content of README.md file
    """
    if not README_PATH.exists():
        return DESCRIPTION

    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()

    return long_description


def setup():
    """
    Main setup function that configures the Python package build process.
    """
    # Read requirements
    install_requires = read_requirements(REQUIREMENTS_PATH)
    dev_requires = read_requirements(DEV_REQUIREMENTS_PATH)

    # Read long description
    long_description = read_long_description()

    # Ensure PyQt6 is in dependencies if not already included
    if not any(req.startswith("PyQt6") for req in install_requires):
        install_requires.append("PyQt6>=6.6.1")

    # Add core scientific libraries if not already included
    core_packages = [
        "numpy>=1.26.3",
        "scipy>=1.11.4",
        "pandas>=2.1.4",
        "statsmodels>=0.14.1",
        "numba>=0.59.0",
        "matplotlib>=3.8.2",
    ]

    for package in core_packages:
        if not any(req.startswith(package.split(">=")[0]) for req in install_requires):
            install_requires.append(package)

    # Configure setup
    setuptools.setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        license=LICENSE,
        python_requires=PYTHON_REQUIRES,
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
        install_requires=install_requires,
        extras_require={
            "dev": dev_requires,
        },
        include_package_data=True,
        package_data={
            "mfe_web": ["resources/*", "ui/*"],
        },
        entry_points={
            "console_scripts": [
                "mfe-gui=mfe_web.ui.main:main",
            ],
            "gui_scripts": [
                "mfe-gui-app=mfe_web.ui.main:main",
            ],
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Financial and Insurance Industry",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Office/Business :: Financial",
            "Operating System :: OS Independent",
        ],
        url="https://github.com/bashtage/arch",
        project_urls={
            "Documentation": "https://bashtage.github.io/arch/",
            "Bug Reports": "https://github.com/bashtage/arch/issues",
            "Source": "https://github.com/bashtage/arch",
        },
    )


if __name__ == "__main__":
    setup()