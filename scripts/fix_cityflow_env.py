#!/usr/bin/env python
# scripts/fix_cityflow_env.py
# Fix for the CityFlow environment adapter

import os
import sys
import shutil
from datetime import datetime

def main():
    """Fix the CityFlow adapter to handle action mapping correctly"""
    print("=== CITYFLOW ADAPTER FIX ===")
    
    # Find the CityFlow environment file
    env_file = 'ragen/env/traffic_control/utils/cityflow_env.py'
    if not os.path.exists(env_file):
        env_file = 'env/traffic_control/utils/cityflow_env.py'
        if not os.path.exists(env_file):
            print("Could not find CityFlow adapter file")
            return False
    
    print(f"Found CityFlow adapter at: {env_file}")
    
    # Create a backup
    backup_path = f"{env_file}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(env_file, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Look for the step method
    if 'def step(self, action):' in content:
        print("Found step method - applying fix")
        
        # Find the step method and add the action mapping
        modified_content = content.replace(
            'def step(self, action):',
            'def step(self, action):\n        # Map actions 1-4 from TrafficControlEnv to 0-3 for CityFlow\n        if 1 <= action <= 4:\n            action = action - 1\n            print(f"Mapped action {action+1} to CityFlow action {action}")\n'
        )
        
        # Write the modified content back to the file
        with open(env_file, 'w') as f:
            f.write(modified_content)
        
        print(f"Successfully patched {env_file}")
        print("The CityFlow adapter now correctly maps actions 1-4 to 0-3")
        return True
    else:
        print("Could not find the step method in the CityFlow adapter file")
        print("Please check the file and manually add the following code at the beginning of the step method:")
        print("""
        # Map actions 1-4 from TrafficControlEnv to 0-3 for CityFlow
        if 1 <= action <= 4:
            action = action - 1
            print(f"Mapped action {action+1} to CityFlow action {action}")
        """)
        return False

def create_test_script():
    """Create a simple test script for the fixed environment"""
    test_file = 'scripts/test_cityflow_fix.py'
    
    content = """#!/usr/bin/env python
# Test script for the fixed CityFlow adapter

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_control.env import TrafficControlEnv

def main():
    print("=== TESTING FIXED CITYFLOW ADAPTER ===")
    
    # Make sure data directory exists
    os.makedirs('./data/traffic', exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = TrafficControlEnv(
        path_to_work_directory='./data/traffic',
        max_steps=20,
        num_intersections=1
    )
    
    # Reset environment
    print("Resetting environment...")
    obs = env.reset()
    print("Environment reset successful")
    
    # Try action 0 (should be invalid)
    print("\\nTesting invalid action 0:")
    obs, reward, done, info = env.step(0)
    print(f"Reward: {reward}")
    print(f"Valid: {info.get('action_is_valid', 'Unknown')}")
    
    # Try actions 1-4 (should be valid now)
    for action in range(1, 5):
        print(f"\\nTesting action {action}:")
        try:
            obs, reward, done, info = env.step(action)
            print(f"SUCCESS! Action {action} worked without crashing")
            print(f"Reward: {reward}")
            print(f"Valid: {info.get('action_is_valid', 'Unknown')}")
            
            if done:
                print("Environment signaled completion")
                break
        except Exception as e:
            print(f"ERROR: Action {action} caused an exception: {e}")
        
        time.sleep(0.5)
    
    print("\\nTest completed")

if __name__ == "__main__":
    main()
"""
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    os.chmod(test_file, 0o755)  # Make it executable
    print(f"Created test script at {test_file}")
    print("Run it with: python scripts/test_cityflow_fix.py")

if __name__ == "__main__":
    main()
    create_test_script() 