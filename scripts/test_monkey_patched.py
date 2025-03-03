#!/usr/bin/env python
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
        
        print(f"\nStep {step+1} - Taking action {action} ({env.ACTION_LOOKUP.get(action, 'Unknown')})")
        
        obs, reward, done, info = env.step(action)
        
        # Print basic info
        print(f"Reward: {reward}")
        valid = info.get('action_is_valid', False)
        print(f"Action validity: {'✓ Valid' if valid else '✗ Invalid'}")
        
        # Show observation excerpt
        if isinstance(obs, str):
            lines = obs.split('\n')
            print("\nObservation (excerpt):")
            for i, line in enumerate(lines[:10]):
                print(f"  {line[:80]}")
            if len(lines) > 10:
                print(f"  ... ({len(lines)-10} more lines)")
        
        if done:
            print("\nEnvironment signaled completion")
            break
        
        time.sleep(0.5)  # Brief pause between steps
    
    # Close environment
    if hasattr(env, 'close'):
        env.close()
    
    print("\nTest completed successfully")

if __name__ == "__main__":
    main()
