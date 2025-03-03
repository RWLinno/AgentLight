#!/usr/bin/env python
# scripts/fix_traffic_env.py
# This script creates a patched version of the TrafficControlEnv class that fixes the action indexing issue

import os
import sys
import shutil
import re
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the original file"""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
        return True
    return False

def manually_fix_env(env_file):
    """
    Display instructions for manually fixing the environment
    """
    print("\n=== MANUAL FIX INSTRUCTIONS ===")
    print(f"Please edit the file: {env_file}")
    print("\n1. Find the method that handles actions (likely named 'step' or similar)")
    print("2. Find where the method passes the action to the CityFlow engine")
    print("3. Add the following code before that call:")
    
    print("""
    # Map actions 1-4 to CityFlow indexing (0-3)
    cityflow_action = action
    if 1 <= action <= 4:
        cityflow_action = action - 1
    """)
    
    print("\n4. Replace any call that passes 'action' to CityFlow with 'cityflow_action' instead")
    print("   For example, change:")
    print("   obs, reward, done, info = self.cityflow_env.step(action)")
    print("   to:")
    print("   obs, reward, done, info = self.cityflow_env.step(cityflow_action)")
    
    print("\nAfter making these changes, you should be able to use actions 1-4 without segmentation faults.")
    return False

def patch_environment_code():
    """
    Create a patched version of the TrafficControlEnv class that fixes the action indexing issue
    """
    # Find the environment file
    env_file = 'env/traffic_control/env.py'
    if not os.path.exists(env_file):
        env_file = 'ragen/env/traffic_control/env.py'
        if not os.path.exists(env_file):
            print("Could not find TrafficControlEnv implementation file")
            return False
    
    print(f"Found environment file at: {env_file}")
    
    # Create a backup
    if not backup_file(env_file):
        print("Failed to create backup, aborting")
        return False
    
    # Read the existing file
    with open(env_file, 'r') as f:
        content = f.read()
    
    print("Analyzing environment code...")
    
    # Look for ALL ways the environment might pass actions to CityFlow
    patterns = [
        r'(self\.cityflow_env\.step\s*\()(\s*action\s*\))',  # Direct call to cityflow_env.step
        r'(CityFlowEnv.*?step\s*\()(\s*action\s*\))',  # Call to CityFlowEnv.step
        r'(cityflow_env.*?step\s*\()(\s*action\s*\))',  # Any cityflow_env variable
        r'(_step\s*\()(\s*action\s*\))'  # Potential internal _step method
    ]
    
    found_match = False
    fixed_content = content
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, fixed_content, re.IGNORECASE | re.DOTALL))
        if matches:
            print(f"Found {len(matches)} potential places to patch")
            found_match = True
            
            # Add a global action mapping at the class level
            class_def_pattern = r'(class\s+TrafficControlEnv.*?:.*?\n)'
            if re.search(class_def_pattern, fixed_content, re.DOTALL):
                fixed_content = re.sub(
                    class_def_pattern,
                    r'\1\n    # Map between ACTION_LOOKUP actions (1-4) and CityFlow actions (0-3)\n    def _map_action_to_cityflow(self, action):\n        cityflow_action = action\n        if 1 <= action <= 4:\n            cityflow_action = action - 1\n        return cityflow_action\n\n',
                    fixed_content,
                    count=1,
                    flags=re.DOTALL
                )
            
            # Replace each match with the mapped action
            for match in matches:
                orig = match.group(0)
                replacement = f"{match.group(1)}self._map_action_to_cityflow({match.group(2).strip()})"
                fixed_content = fixed_content.replace(orig, replacement)
    
    if found_match:
        # Write the patched file
        with open(env_file, 'w') as f:
            f.write(fixed_content)
        
        print(f"Successfully patched {env_file}")
        print("The environment now maps actions 1-4 to CityFlow's expected 0-3 index range")
        return True
    else:
        print("Could not automatically locate where to patch the code.")
        return manually_fix_env(env_file)

def create_direct_fix_script():
    """Create a direct fix script that can be used with exec to avoid regex issues"""
    fix_script = 'scripts/direct_fix.py'
    
    content = """#!/usr/bin/env python
# Direct fix for TrafficControlEnv
# This script directly modifies the environment without relying on regex

import os
import sys
import shutil
from datetime import datetime

def main():
    # Find the environment file
    env_file = 'env/traffic_control/env.py'
    if not os.path.exists(env_file):
        env_file = 'ragen/env/traffic_control/env.py'
        if not os.path.exists(env_file):
            print("Could not find TrafficControlEnv implementation file")
            return False
    
    print(f"Found environment file at: {env_file}")
    
    # Create a backup
    backup_path = f"{env_file}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(env_file, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Import the environment class to inspect it
    sys.path.append('.')
    
    try:
        from env.traffic_control.env import TrafficControlEnv
        env_class = TrafficControlEnv
        print("Successfully imported TrafficControlEnv")
    except ImportError:
        try:
            from ragen.env.traffic_control.env import TrafficControlEnv
            env_class = TrafficControlEnv
            print("Successfully imported TrafficControlEnv from ragen")
        except ImportError:
            print("Could not import TrafficControlEnv, falling back to manual patching")
            return False
    
    # Add the mapping method to the class
    def _map_action_to_cityflow(self, action):
        cityflow_action = action
        if 1 <= action <= 4:
            cityflow_action = action - 1
        return cityflow_action
    
    # Add the method to the class
    TrafficControlEnv._map_action_to_cityflow = _map_action_to_cityflow
    
    # Monkey patch the step method
    original_step = TrafficControlEnv.step
    
    def patched_step(self, action):
        # Use the original action for everything except cityflow calls
        # When passing to cityflow, map it
        cityflow_action = self._map_action_to_cityflow(action)
        
        # Store the original method
        original_cityflow_step = None
        if hasattr(self, 'cityflow_env') and hasattr(self.cityflow_env, 'step'):
            original_cityflow_step = self.cityflow_env.step
            
            # Replace it with our version that uses the mapped action
            def patched_cityflow_step(act):
                return original_cityflow_step(cityflow_action)
            
            self.cityflow_env.step = patched_cityflow_step
        
        # Call the original step
        result = original_step(self, action)
        
        # Restore the original cityflow step method
        if original_cityflow_step:
            self.cityflow_env.step = original_cityflow_step
        
        return result
    
    # Replace the step method
    TrafficControlEnv.step = patched_step
    
    print("Successfully patched TrafficControlEnv.step method")
    print("The environment now maps actions 1-4 to CityFlow's expected 0-3 index range")
    
    # Write the fix to the code
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Add our mapping method
    if "def _map_action_to_cityflow" not in content:
        # Find the class definition line
        class_pattern = "class TrafficControlEnv"
        if class_pattern in content:
            lines = content.split('\\n')
            class_index = -1
            for i, line in enumerate(lines):
                if class_pattern in line:
                    class_index = i
                    break
            
            if class_index >= 0:
                # Find the end of class definition and start of first method
                for i in range(class_index+1, len(lines)):
                    if lines[i].strip().startswith('def '):
                        # Insert our method before the first method
                        method_code = [
                            "    # Map between ACTION_LOOKUP actions (1-4) and CityFlow actions (0-3)",
                            "    def _map_action_to_cityflow(self, action):",
                            "        cityflow_action = action",
                            "        if 1 <= action <= 4:",
                            "            cityflow_action = action - 1",
                            "        return cityflow_action",
                            ""
                        ]
                        lines = lines[:i] + method_code + lines[i:]
                        break
                
                content = '\\n'.join(lines)
                
                # Now look for the step method
                step_pattern = "    def step(self, action):"
                if step_pattern in content:
                    # Find all the places where cityflow_env.step is called
                    cityflow_step_pattern = "cityflow_env.step(action)"
                    content = content.replace(cityflow_step_pattern, "cityflow_env.step(self._map_action_to_cityflow(action))")
                    
                    # Write updated file
                    with open(env_file, 'w') as f:
                        f.write(content)
                    
                    print("Successfully patched cityflow_env.step calls in the code")
                    return True
    
    print("Could not update the code directly.")
    print("However, the runtime monkey patch is active for this session.")
    return True

if __name__ == "__main__":
    main()
"""
    
    with open(fix_script, 'w') as f:
        f.write(content)
    
    os.chmod(fix_script, 0o755)  # Make it executable
    print(f"Created direct fix script at {fix_script}")
    print("Run it with: python scripts/direct_fix.py")

def create_monkey_patch_test():
    """Create a test script that uses monkey patching to fix the environment"""
    test_file = 'scripts/test_monkey_patched.py'
    
    content = """#!/usr/bin/env python
# This script uses monkey patching to fix the TrafficControlEnv

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment
from env.traffic_control.env import TrafficControlEnv

# Monkey patch the environment
def _map_action_to_cityflow(self, action):
    #Map from logical actions (1-4) to CityFlow actions (0-3)
    cityflow_action = action
    if 1 <= action <= 4:
        cityflow_action = action - 1
    return cityflow_action

# Add the mapping method
TrafficControlEnv._map_action_to_cityflow = _map_action_to_cityflow

# Store the original step method
original_step = TrafficControlEnv.step

# Create a patched step method
def patched_step(self, action):
    print(f"Taking action {action}, mapped to {self._map_action_to_cityflow(action)} for CityFlow")
    
    # If environment has cityflow_env, patch it temporarily
    if hasattr(self, 'cityflow_env') and hasattr(self.cityflow_env, 'step'):
        # Store original cityflow step
        original_cityflow_step = self.cityflow_env.step
        
        # Create patched cityflow step
        def patched_cityflow_step(act):
            # Always use our mapped action
            mapped_action = self._map_action_to_cityflow(action)
            print(f"CityFlow using mapped action: {mapped_action}")
            return original_cityflow_step(mapped_action)
        
        # Apply patch
        self.cityflow_env.step = patched_cityflow_step
        
        # Call original step
        result = original_step(self, action)
        
        # Restore original cityflow step
        self.cityflow_env.step = original_cityflow_step
        
        return result
    else:
        # Just map the action directly
        mapped_action = self._map_action_to_cityflow(action)
        return original_step(self, mapped_action)

# Apply the patch
TrafficControlEnv.step = patched_step

print("TrafficControlEnv has been monkey patched to fix action mapping")

def main():
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
    
    # Test each valid action (1-4)
    for step in range(8):  # Take 8 steps to see multiple actions
        # Cycle through valid actions 1-4
        action = (step % 4) + 1  # This gives actions 1,2,3,4,1,2,3,4
        
        print(f"\\nStep {step+1} - Taking action {action} ({env.ACTION_LOOKUP.get(action, 'Unknown')})")
        
        obs, reward, done, info = env.step(action)
        
        # Print basic info
        print(f"Reward: {reward}")
        valid = info.get('action_is_valid', False)
        print(f"Action validity: {'✓ Valid' if valid else '✗ Invalid'}")
        
        # Show observation excerpt
        if isinstance(obs, str):
            lines = obs.split('\\n')
            print("\\nObservation (excerpt):")
            for i, line in enumerate(lines[:10]):
                print(f"  {line[:80]}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")
        
        if done:
            print("\\nEnvironment signaled completion")
            break
        
        time.sleep(0.5)  # Brief pause between steps
    
    # Close environment
    if hasattr(env, 'close'):
        env.close()
    
    print("\\nTest completed successfully")

if __name__ == "__main__":
    main()
"""
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    os.chmod(test_file, 0o755)  # Make it executable
    print(f"Created monkey patch test script at {test_file}")
    print("Run it with: python scripts/test_monkey_patched.py")

def create_simple_test():
    """Create a simple test script that uses the patched environment"""
    test_file = 'scripts/test_patched_env.py'
    
    content = """#!/usr/bin/env python
# This script tests the patched TrafficControlEnv that fixes the action indexing issue

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_control.env import TrafficControlEnv

def main():
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
    
    # Test each valid action (1-4)
    for step in range(8):  # Take 8 steps to see multiple actions
        # Cycle through valid actions 1-4
        action = (step % 4) + 1  # This gives actions 1,2,3,4,1,2,3,4
        
        print(f"\\nStep {step+1} - Taking action {action} ({env.ACTION_LOOKUP.get(action, 'Unknown')})")
        
        obs, reward, done, info = env.step(action)
        
        # Print basic info
        print(f"Reward: {reward}")
        valid = info.get('action_is_valid', False)
        print(f"Action validity: {'✓ Valid' if valid else '✗ Invalid'}")
        
        # Show observation excerpt
        if isinstance(obs, str):
            lines = obs.split('\\n')
            print("\\nObservation (excerpt):")
            for i, line in enumerate(lines[:10]):
                print(f"  {line[:80]}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")
        
        if done:
            print("\\nEnvironment signaled completion")
            break
        
        time.sleep(0.5)  # Brief pause between steps
    
    # Close environment
    if hasattr(env, 'close'):
        env.close()
    
    print("\\nTest completed successfully")

if __name__ == "__main__":
    main()
"""
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    os.chmod(test_file, 0o755)  # Make it executable
    print(f"Created test script at {test_file}")
    print("Run it with: python scripts/test_patched_env.py")

def main():
    """Main function to fix the traffic environment"""
    print("=== ENHANCED TRAFFIC ENVIRONMENT FIXER ===")
    print("This script patches the TrafficControlEnv to fix the action indexing issue")
    
    # Try the regex-based approach first
    if patch_environment_code():
        create_simple_test()
        print("\nThe environment has been successfully patched!")
        print("You can now use actions 1-4 without segmentation faults.")
        print("Try running the test script: python scripts/test_patched_env.py")
    else:
        # If that fails, create alternative scripts
        print("\nCreating alternative fix methods...")
        create_direct_fix_script()
        create_monkey_patch_test()
        print("\nTwo alternative fix methods have been created:")
        print("1. Direct code modification: python scripts/direct_fix.py")
        print("2. Runtime monkey patching: python scripts/test_monkey_patched.py")
        print("\nTry method #2 first - it doesn't modify any code but should fix the issue for testing.")

if __name__ == "__main__":
    main() 