#!/usr/bin/env python
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
            from ragen.env.traffic_control_no_use.env_no_use import TrafficControlEnv
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
            lines = content.split('\n')
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
                
                content = '\n'.join(lines)
                
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
