#!/usr/bin/env python3
"""
Script to fix CityFlow dependencies by updating the C++ standard library.
This addresses the GLIBCXX_3.4.30 missing issue.
"""

import os
import sys
import subprocess
import platform

def print_banner(text):
    """Print a section banner"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def run_command(command, exit_on_error=True):
    """Run a shell command and return its output"""
    print(f"\n> {' '.join(command)}")
    try:
        result = subprocess.run(command, 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        if exit_on_error:
            print("Exiting due to error.")
            sys.exit(1)
        return None

def is_conda_environment():
    """Check if running in a conda environment"""
    return 'CONDA_PREFIX' in os.environ

def fix_libstdcxx():
    """Update libstdcxx-ng to get a newer GLIBCXX version"""
    print_banner("Updating C++ Standard Library")
    
    if not is_conda_environment():
        print("Error: Not running in a conda environment.")
        print("This script is designed to fix conda environments.")
        return False
    
    conda_prefix = os.environ['CONDA_PREFIX']
    print(f"Conda environment: {conda_prefix}")
    
    # First, try updating libstdcxx-ng from conda-forge
    run_command(['conda', 'install', '-y', '-c', 'conda-forge', 'libstdcxx-ng'])
    
    # Check if we now have the required GLIBCXX version
    try:
        libstdc_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
        output = run_command(['strings', libstdc_path, '|', 'grep', 'GLIBCXX'], exit_on_error=False)
        if output and 'GLIBCXX_3.4.30' in output:
            print("\nSuccess! GLIBCXX_3.4.30 is now available.")
            return True
    except Exception as e:
        print(f"Error checking libstdc++: {e}")
    
    # If we're here, we still don't have the required version
    print("\nThe required GLIBCXX_3.4.30 version is still not available.")
    print("Trying alternative methods...")
    
    # Try installing gcc/g++ which should bring in a newer libstdc++
    run_command(['conda', 'install', '-y', '-c', 'conda-forge', 'gcc', 'gxx'])
    
    return False

def reinstall_cityflow():
    """Reinstall CityFlow package"""
    print_banner("Reinstalling CityFlow")
    
    # First uninstall current version
    run_command(['pip', 'uninstall', '-y', 'cityflow'])
    
    # Try to reinstall from pip
    run_command(['pip', 'install', 'cityflow'])
    
    return True

def test_cityflow_import():
    """Test if CityFlow can now be imported"""
    print_banner("Testing CityFlow Import")
    
    try:
        import cityflow
        print("✓ Success! CityFlow can now be imported.")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CityFlow: {e}")
        return False

def create_env_script():
    """Create a wrapper script to set LD_LIBRARY_PATH if needed"""
    print_banner("Creating Environment Setup Script")
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("Not in a conda environment. Skipping wrapper script creation.")
        return
    
    script_content = f"""#!/bin/bash
# This script sets up the environment for CityFlow to work with Ray
# Source this script before running your Ray application

# Add newer libstdc++ to LD_LIBRARY_PATH if it exists
if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
fi

# You can add other libraries if needed
# export LD_LIBRARY_PATH="/path/to/other/libs:$LD_LIBRARY_PATH"

# Print environment for verification
echo "Environment configured for CityFlow with Ray"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
"""
    
    with open('setup_cityflow_env.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('setup_cityflow_env.sh', 0o755)
    print("Created setup_cityflow_env.sh script")
    print("Source this script before running your Ray application:")
    print("  source ./setup_cityflow_env.sh")

def main():
    print("=" * 70)
    print("CityFlow Dependency Fixer")
    print("This script will attempt to fix the missing GLIBCXX_3.4.30 issue")
    print("=" * 70)
    
    # First try to update the C++ standard library
    fix_libstdcxx()
    
    # Reinstall CityFlow
    reinstall_cityflow()
    
    # Test if CityFlow can now be imported
    success = test_cityflow_import()
    
    # Create environment setup script as a fallback
    if not success:
        create_env_script()
        print("\nIf you're still having issues, you might need to install a system-wide")
        print("libstdc++ package with apt or your system's package manager:")
        print("  sudo apt-get update")
        print("  sudo apt-get install libstdc++6")
    
    print("\nScript completed!")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 