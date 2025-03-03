#!/usr/bin/env python
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
    print("\nTesting invalid action 0:")
    obs, reward, done, info = env.step(0)
    print(f"Reward: {reward}")
    print(f"Valid: {info.get('action_is_valid', 'Unknown')}")
    
    # Try actions 1-4 (should be valid now)
    for action in range(1, 5):
        print(f"\nTesting action {action}:")
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
    
    print("\nTest completed")

if __name__ == "__main__":
    main()
