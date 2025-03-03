#!/usr/bin/env python3
"""
Test script to verify CityFlow installation and functionality.
This script attempts to import CityFlow and run a basic simulation
without using Ray, to isolate dependency issues.
"""

import os
import sys
import json
from pathlib import Path

def print_system_info():
    """Print system and Python environment information"""
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Print library path info
    print("\n=== Library Paths ===")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Print conda environment info if available
    if 'CONDA_PREFIX' in os.environ:
        print(f"Conda environment: {os.environ['CONDA_PREFIX']}")

def test_cityflow_import():
    """Test importing the CityFlow module"""
    print("\n=== Testing CityFlow Import ===")
    try:
        import cityflow
        print("✓ Successfully imported CityFlow")
        return cityflow
    except ImportError as e:
        print(f"✗ Failed to import CityFlow: {e}")
        
        # Check for libstdc++ issues specifically
        if 'GLIBCXX' in str(e):
            print("\nThis appears to be a C++ standard library version issue.")
            print("Possible solutions:")
            print("1. Update libstdc++ in your conda environment:")
            print("   conda install -c conda-forge libstdcxx-ng")
            print("2. Or install a compatible version of CityFlow")
        return None


def test_cityflow_simulation(cityflow_module, config_path):
    """Test running a basic CityFlow simulation"""
    print("\n=== Testing CityFlow Simulation ===")
    try:
        # Initialize CityFlow
        eng = cityflow_module.Engine(config_path)
        print("✓ Successfully created CityFlow Engine")
        
        # Run a few simulation steps
        for i in range(10):
            eng.next_step()
        print("✓ Successfully ran simulation steps")
        
        return True
    except Exception as e:
        print(f"✗ Error during simulation: {e}")
        return False

def main():
    print("=== CityFlow Test Script ===")
    
    # Print system information
    print_system_info()
    
    # Test importing CityFlow
    cityflow_module = test_cityflow_import()
    if cityflow_module is None:
        return 1
    
    # Create test configuration
    data_dir = "./data/traffic"
    #config_path = create_test_config(data_dir)
    
    # Test running a simulation
    success = test_cityflow_simulation(cityflow_module, "/home/weilinruan/AgentLight/data/traffic/config.json")
    
    if success:
        print("\n=== All CityFlow tests passed! ===")
        print("CityFlow is working correctly outside of Ray.")
        print("The issue might be related to how Ray is interacting with CityFlow.")
    else:
        print("\n=== CityFlow tests failed ===")
        print("There appears to be an issue with the CityFlow installation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 