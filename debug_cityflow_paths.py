#!/usr/bin/env python3
"""
Debug script to verify the paths that CityFlow is using.
"""
import os
import sys
import json

def print_file_info(filepath):
    """Print information about a file"""
    print(f"Checking file: {filepath}")
    print(f"  Absolute path: {os.path.abspath(filepath)}")
    if os.path.exists(filepath):
        print(f"  ✓ File exists")
        print(f"  Size: {os.path.getsize(filepath)} bytes")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                print(f"  First 100 chars: {content[:100]}...")
        except Exception as e:
            print(f"  Error reading file: {e}")
    else:
        print(f"  ✗ File DOES NOT exist")
        parent_dir = os.path.dirname(filepath)
        if os.path.exists(parent_dir):
            print(f"  Parent directory exists: {parent_dir}")
            print(f"  Files in parent directory: {os.listdir(parent_dir)}")
        else:
            print(f"  Parent directory does not exist: {parent_dir}")

def main():
    """Test loading roadnet and flow files"""
    print("\n=== CityFlow Path Debug ===")
    print(f"Current working directory: {os.getcwd()}")
    
    work_dir = "./data/traffic/"
    config_path = os.path.join(work_dir, "config.json")
    roadnet_path = os.path.join(work_dir, "roadnet.json")
    flow_path = os.path.join(work_dir, "flow.json")
    
    print("\nChecking work directory...")
    if os.path.exists(work_dir):
        print(f"✓ Work directory exists: {work_dir}")
        print(f"Files in work directory: {os.listdir(work_dir)}")
    else:
        print(f"✗ Work directory DOES NOT exist: {work_dir}")
        print("Creating work directory...")
        os.makedirs(work_dir, exist_ok=True)
    
    print("\nChecking config file...")
    print_file_info(config_path)
    
    print("\nChecking roadnet file...")
    print_file_info(roadnet_path)
    
    print("\nChecking flow file...")
    print_file_info(flow_path)

    print("\nAttempting to load CityFlow...")
    try:
        import cityflow
        print("✓ Successfully imported CityFlow")
    except ImportError as e:
        print(f"✗ Failed to import CityFlow: {e}")
        return 1
    
    print("Attempting to create Engine with config: " + config_path)
    try:
        eng = cityflow.Engine(config_path)
        print("✓ Successfully created CityFlow Engine")
        
        # Try running a few steps
        print("Running simulation steps...")
        for i in range(10):
            eng.next_step()
        print("✓ Successfully ran simulation steps")
    except Exception as e:
        print(f"✗ Failed to create/run CityFlow Engine: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
