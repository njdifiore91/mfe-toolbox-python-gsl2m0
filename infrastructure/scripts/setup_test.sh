#!/usr/bin/env python3
"""
setup_venv.py - Development Environment Setup for MFE Toolbox

This script sets up a development environment for the MFE (MATLAB Financial Econometrics) Toolbox,
a comprehensive suite of Python modules for financial time series modeling and econometric analysis.

The script performs the following tasks:
- Creates a Python 3.12 virtual environment
- Installs test dependencies and other requirements
- Configures pytest settings 
- Sets up logging
- Handles cleanup on failure

Usage:
  python setup_venv.py [--clean]
"""

import os
import sys
import venv
import subprocess
import logging
import argparse
import shutil
from pathlib import Path

# Configuration based on JSON specification
CONFIG = {
    'pythonVersion': '3.12',
    'virtualEnvironment': {
        'directory': '.venv',
        'cleanInstall': True
    },
    'testDependencies': {
        'pytest': '7.4.3',
        'pytest-asyncio': '0.21.1',
        'pytest-cov': '4.1.0',
        'pytest-benchmark': '4.0.0',
        'hypothesis': '6.92.1'
    },
    'requirementsFiles': [
        'src/backend/requirements.txt',
        'src/web/requirements.txt'
    ],
    'testConfiguration': {
        'configFile': 'pytest.ini',
        'defaultSettings': {
            'testpaths': ['tests'],
            'pythonFiles': 'test_*.py',
            'pythonClasses': 'Test*',
            'pythonFunctions': 'test_*',
            'additionalOptions': '--strict-markers -v --cov=mfe --cov-report=term-missing'
        }
    },
    'logging': {
        'enabled': True,
        'levels': ['info', 'error']
    },
    'cleanup': {
        'removeVenvOnFailure': True,
        'deactivateVenvOnFailure': True
    }
}

# Core dependencies based on technical specification
CORE_DEPENDENCIES = [
    'numpy>=1.26.3',
    'scipy>=1.11.4',
    'pandas>=2.1.4',
    'statsmodels>=0.14.1',
    'numba>=0.59.0',
    'pyqt6>=6.6.1',
    'matplotlib>=3.7.0'
]

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Setup development environment for MFE Toolbox')
    parser.add_argument('--clean', action='store_true', help='Force clean installation of virtual environment')
    return parser.parse_args()

def setup_logging():
    """Configure logging based on settings"""
    if CONFIG['logging']['enabled']:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.disable(logging.CRITICAL)
    
    return logging.getLogger(__name__)

def check_python_version(logger):
    """Check if current Python version meets requirements"""
    python_version = sys.version_info
    required_version = tuple(map(int, CONFIG['pythonVersion'].split('.')))
    
    if python_version < required_version:
        logger.error(f"Python {CONFIG['pythonVersion']} or higher is required. Found {sys.version}")
        return False
    
    logger.info(f"Python version check passed: {sys.version}")
    return True

def create_virtual_environment(logger, args):
    """Create a virtual environment"""
    venv_dir = Path(CONFIG['virtualEnvironment']['directory'])
    
    # Clean existing venv if requested via config or command line
    if (CONFIG['virtualEnvironment']['cleanInstall'] or args.clean) and venv_dir.exists():
        logger.info(f"Removing existing virtual environment at {venv_dir}")
        shutil.rmtree(venv_dir)
    
    if not venv_dir.exists():
        logger.info(f"Creating virtual environment at {venv_dir}")
        venv.create(venv_dir, with_pip=True)
        return True
    else:
        logger.info(f"Using existing virtual environment at {venv_dir}")
        return False

def get_venv_python(venv_dir):
    """Get path to Python executable in virtual environment"""
    if sys.platform == 'win32':
        return str(Path(venv_dir) / 'Scripts' / 'python.exe')
    else:
        return str(Path(venv_dir) / 'bin' / 'python')

def install_requirements(logger, venv_python):
    """Install required dependencies"""
    logger.info("Installing test dependencies")
    
    # Install test dependencies
    test_deps = [f"{dep}=={ver}" for dep, ver in CONFIG['testDependencies'].items()]
    cmd = [venv_python, "-m", "pip", "install"] + test_deps
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Install requirements from files
    for req_file in CONFIG['requirementsFiles']:
        req_path = Path(req_file)
        if req_path.exists():
            logger.info(f"Installing dependencies from {req_file}")
            subprocess.run([venv_python, "-m", "pip", "install", "-r", str(req_path)], check=True)
        else:
            logger.warning(f"Requirements file not found: {req_file}")
    
    # Install core dependencies from technical specification
    logger.info("Installing core dependencies for MFE Toolbox")
    cmd = [venv_python, "-m", "pip", "install"] + CORE_DEPENDENCIES
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Install the package in development mode if setup.py or pyproject.toml exists
    if Path('setup.py').exists() or Path('pyproject.toml').exists():
        logger.info("Installing MFE Toolbox in development mode")
        subprocess.run([venv_python, "-m", "pip", "install", "-e", "."], check=True)
    
    logger.info("All dependencies installed successfully")

def generate_pytest_config(logger):
    """Generate pytest configuration file"""
    config_path = Path(CONFIG['testConfiguration']['configFile'])
    
    logger.info(f"Generating pytest configuration at {config_path}")
    
    settings = CONFIG['testConfiguration']['defaultSettings']
    
    with open(config_path, 'w') as f:
        f.write("[pytest]\n")
        f.write(f"testpaths = {' '.join(settings['testpaths'])}\n")
        f.write(f"python_files = {settings['pythonFiles']}\n")
        f.write(f"python_classes = {settings['pythonClasses']}\n")
        f.write(f"python_functions = {settings['pythonFunctions']}\n")
        f.write(f"addopts = {settings['additionalOptions']}\n")
    
    logger.info(f"Pytest configuration generated at {config_path}")

def cleanup_on_failure(logger, created_venv):
    """Clean up on failure"""
    if CONFIG['cleanup']['removeVenvOnFailure'] and created_venv:
        venv_dir = Path(CONFIG['virtualEnvironment']['directory'])
        if venv_dir.exists():
            logger.info(f"Cleaning up: removing virtual environment at {venv_dir}")
            shutil.rmtree(venv_dir)

def main():
    """Main function to set up the development environment"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("Starting MFE Toolbox development environment setup")
    logger.info("This will set up a Python environment for financial econometrics analysis")
    logger.info(f"MFE Toolbox uses Python {CONFIG['pythonVersion']} with NumPy, SciPy, Pandas, Statsmodels, and Numba")
    
    created_venv = False
    
    try:
        # Check Python version
        if not check_python_version(logger):
            return 1
        
        # Create virtual environment
        created_venv = create_virtual_environment(logger, args)
        
        # Get path to venv Python
        venv_python = get_venv_python(CONFIG['virtualEnvironment']['directory'])
        
        # Install dependencies
        install_requirements(logger, venv_python)
        
        # Generate pytest.ini
        generate_pytest_config(logger)
        
        logger.info("MFE Toolbox development environment setup complete!")
        logger.info(f"To activate the virtual environment:")
        if sys.platform == 'win32':
            logger.info(f"    {CONFIG['virtualEnvironment']['directory']}\\Scripts\\activate")
        else:
            logger.info(f"    source {CONFIG['virtualEnvironment']['directory']}/bin/activate")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        
        if CONFIG['cleanup']['deactivateVenvOnFailure']:
            logger.info("Deactivating virtual environment")
        
        cleanup_on_failure(logger, created_venv)
        return 1

if __name__ == "__main__":
    sys.exit(main())