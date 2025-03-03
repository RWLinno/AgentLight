#!/usr/bin/env python3
"""
Fix configuration parameters for traffic signal control training.
"""
import os
import sys

def fix_config():
    """Add missing configuration parameters to the default configs."""
    try:
        # Find the base trainer config file
        config_dir = "./verl/configs/trainer"
        if not os.path.exists(config_dir):
            print(f"Config directory not found: {config_dir}")
            return False
        
        # Look for _base_.yaml or similar files
        base_configs = [f for f in os.listdir(config_dir) if "_base_" in f or "defaults" in f]
        if not base_configs:
            print("No base config files found.")
            return False
        
        # Add the ref_update_steps parameter to each base config
        for config_file in base_configs:
            file_path = os.path.join(config_dir, config_file)
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if parameter already exists
            if "ref_update_steps:" not in content:
                # Find a good place to insert the parameter
                if "save_freq:" in content:
                    new_content = content.replace(
                        "save_freq:", 
                        "ref_update_steps: 10  # How often to update reference model\nsave_freq:"
                    )
                    
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    print(f"Added ref_update_steps parameter to {file_path}")
                else:
                    print(f"Could not find a good insertion point in {file_path}")
        
        return True
    except Exception as e:
        print(f"Error fixing config: {e}")
        return False

if __name__ == "__main__":
    success = fix_config()
    if success:
        print("Successfully added missing configuration parameters")
    else:
        print("Failed to add missing configuration parameters") 