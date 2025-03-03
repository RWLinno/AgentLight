import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic.traffic_adapter import TrafficControlEnv
import time

def test_traffic_env():
    """Test the traffic control environment."""
    print("Testing the TrafficControlEnv...")
    
    # Initialize the environment
    env = TrafficControlEnv(
        path_to_log="./log",
        path_to_work_directory="./data/traffic",
        max_steps=10
    )
    
    # Reset the environment
    obs = env.reset()
    print("\nInitial observation:")
    print(obs)
    
    # Run a few steps
    for i in range(5):
        print(f"\nStep {i+1}")
        # Choose a random action (0-3)
        action = i % 4
        print(f"Taking action: {action}")
        
        # Execute the action
        obs, reward, done, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        # Only print the first few lines of observation to save space
        obs_lines = obs.split('\n')
        short_obs = '\n'.join(obs_lines[:10]) + "\n..."
        print(f"Observation: \n{short_obs}")
        
        if done:
            print("Episode finished early")
            break
        
        time.sleep(1)  # Pause to make output readable
    
    print("\nTesting text to action extraction:")
    test_texts = [
        "After analyzing the traffic conditions, I choose signal phase: 2",
        "The best action would be phase 1",
        "I recommend setting the signal to 3"
    ]
    
    for text in test_texts:
        action = env.extract_action(text)
        print(f"Text: '{text}'")
        print(f"Extracted action: {action}")
    
    print("\nTrafficControlEnv test completed.")

if __name__ == "__main__":
    test_traffic_env() 