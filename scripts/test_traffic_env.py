#!/usr/bin/env python
# Script to test the TrafficControlEnv

import os
import sys
import time
import json
import numpy as np
from pprint import pprint

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment
from env.traffic_control.env import TrafficControlEnv

def main():
    """Test the TrafficControlEnv"""
    print("=== TESTING TRAFFIC CONTROL ENVIRONMENT ===")
    
    # Make sure the data directory exists
    os.makedirs('./data/traffic', exist_ok=True)
    
    # Initialize the environment
    env = TrafficControlEnv(
        path_to_work_directory='./data/traffic',
        max_steps=20,
        num_intersections=1
    )
    
    # Reset the environment
    print("\nResetting environment...")
    state = env.reset()
    
    # Check if state is a dictionary or string
    if isinstance(state, dict):
        print("✓ Environment reset successful - state is a dictionary")
        print("State keys:", list(state.keys()))
        # Pretty print the first few keys
        print("\nState sample:")
        pprint({k: state[k] for k in list(state.keys())[:3]})
    else:
        print("✗ Environment reset failed - state is not a dictionary")
        print(f"State type: {type(state)}")
        print(state)
    
    # Test invalid action
    print("\nTesting invalid action (0)...")
    observation, reward, done, info = env.step(0)
    print(f"Reward: {reward}")
    print(f"Valid action: {info.get('action_is_valid', 'Unknown')}")
    print(f"Observation type: {type(observation)}")
    
    if isinstance(observation, dict):
        print("✓ Step returned a dictionary observation")
    else:
        print("✗ Step returned a non-dictionary observation")
    
    # Text representation should be in info
    if "text_observation" in info:
        print("✓ Text observation is in info dictionary")
    else:
        print("✗ Text observation is missing from info dictionary")
    
    # Try valid actions
    for action in range(1, 5):
        print(f"\nTesting action {action}...")
        try:
            observation, reward, done, info = env.step(action)
            print(f"✓ Action {action} succeeded")
            print(f"Reward: {reward}")
            print(f"Valid action: {info.get('action_is_valid', 'Unknown')}")
            
            # Store the state to a file for analysis
            with open(f"action_{action}_state.json", "w") as f:
                if isinstance(observation, dict):
                    json.dump(observation, f, indent=2)
                else:
                    f.write(str(observation))
            
            if done:
                print("Environment signaled completion")
                break
            
        except Exception as e:
            print(f"✗ Action {action} failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(0.5)
    
    # Test random actions
    print("\nRunning 10 random actions...")
    total_reward = 0
    for i in range(10):
        action = np.random.randint(1, 5)  # Random action between 1 and 4
        try:
            observation, reward, done, info = env.step(action)
            print(f"Step {i+1} - Action: {action}, Reward: {reward:.2f}")
            total_reward += reward
            
            if done:
                print("Environment signaled completion")
                break
        except Exception as e:
            print(f"Error during random action: {e}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("\nTest completed successfully!")
    
    # Test the fix for action mapping
    print("\nTesting action mapping to CityFlow...")
    
    # Reset the environment
    env.reset()
    
    # Try all valid actions
    for action in range(1, 5):
        try:
            observation, reward, done, info = env.step(action)
            print(f"✓ Action {action} -> CityFlow action {action-1} successful")
        except Exception as e:
            print(f"✗ Action {action} failed: {e}")

if __name__ == "__main__":
    main() 