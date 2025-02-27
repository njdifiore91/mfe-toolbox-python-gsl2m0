"""
Setup script for building and installing the MFE Toolbox backend package.
This script configures package metadata, dependencies, and build requirements using modern Python packaging tools.
"""

import os
from pathlib import Path

# External imports (setuptools>=69.0.2)
import setuptools  # setuptools version >=69.0.2
from setuptools import setup, find_packages  # setuptools version >=69.0.2

# tomllib is available in Python 3.12 for reading TOML files
import tomllib

# -----------------------------------------------------------------------------
# Internal Imports: Load project metadata from pyproject.toml and __init__.py
# -----------------------------------------------------------------------------

# Define the directory containing this setup.py file
HERE = Path(__file__).parent

# Load project metadata from pyproject.toml
pyproject_file = HERE / "pyproject.toml"
with open(pyproject_file, "rb") as fp:
    pyproject_data = tomllib.load(fp)
# Extract the "project" section which contains metadata fields
project_metadata = pyproject_data.get("project", {})

# Import version information from the package's __init__.py file
from .__init__ import __version__

# -----------------------------------------------------------------------------
# Global Variables: REQUIREMENTS and README
# -----------------------------------------------------------------------------

def read_requirements():
    """
    Reads package requirements from the requirements.txt file.

    Steps:
        - Open requirements.txt
        - Read requirements line by line
        - Strip whitespace and ignore comment lines
        - Return a list of valid requirement strings

    Returns:
        list: List of package requirements with versions.
    """
    requirements = []
    requirements_file = HERE / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as req_file:
            for line in req_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

def read_readme():
    """
    Reads long description from the README.md file.

    Steps:
        - Open README.md file
        - Read the entire content as a string
        - Return the content for use as long_description

    Returns:
        str: Content of README.md file.
    """
    readme_file = HERE / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

REQUIREMENTS = read_requirements()
README = read_readme()

# -----------------------------------------------------------------------------
# Extract Package Metadata from project_metadata
# -----------------------------------------------------------------------------

# Package name, description, and classifiers are read from pyproject.toml
pkg_name = project_metadata.get("name", "mfe")
pkg_description = project_metadata.get("description", "MATLAB Financial Econometrics Toolbox re-implemented in Python")
# Authors are specified as a list of dictionaries; join their names with a comma separator
pkg_authors = project_metadata.get("authors", [])
author_names = ", ".join(author.get("name", "") for author in pkg_authors)
# Classifiers for package indexing and metadata
classifiers = project_metadata.get("classifiers", [])

# -----------------------------------------------------------------------------
# Setup Function: Define the package installation configuration
# -----------------------------------------------------------------------------

setup(
    name=pkg_name,
    version=__version__,
    description=pkg_description,
    long_description=README,
    long_description_content_type="text/markdown",
    author=author_names,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.12",
    classifiers=classifiers,
)