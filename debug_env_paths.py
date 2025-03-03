import os
import sys
from pathlib import Path

def print_path_info(path_str):
    """Print information about a path"""
    path = Path(path_str)
    print(f"Path: {path_str}")
    print(f"  Absolute: {path.absolute()}")
    print(f"  Exists: {path.exists()}")
    if path.exists():
        print(f"  Is file: {path.is_file()}")
        print(f"  Is dir: {path.is_dir()}")
        if path.is_file():
            print(f"  Size: {path.stat().st_size} bytes")

# Print CWD and check common directories
print("\n==== ENVIRONMENT PATH DEBUG ====")
print(f"CWD: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print("\nChecking data directories:")
data_paths = [
    "./data",
    "./data/traffic",
    "./data/traffic/roadnet.json",
    "./data/traffic/flow.json",
]
for p in data_paths:
    print_path_info(p)

print("\nEnvironment Variables:")
for key, value in os.environ.items():
    if "PATH" in key or "DIR" in key:
        print(f"{key}={value}")

print("\n==== DEBUG COMPLETE ====\n")
