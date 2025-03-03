#!/usr/bin/env python3
# This file demonstrates how to use the per-intersection querying functionality
# of the TrafficControlEnv.

import sys
import os
import numpy as np
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from traffic_control.env import TrafficControlEnv

def simulate_llm_response(prompt: str) -> str:
    """
    Simulate an LLM response for the given prompt.
    In a real application, this would be replaced with actual LLM API calls.
    
    Args:
        prompt: Text prompt to send to the LLM
        
    Returns:
        str: Simulated LLM response
    """
    # For demonstration, just return a random phase number
    return f"Based on the traffic conditions, I recommend Phase {np.random.randint(1, 5)}."

def run_per_intersection_demo():
    """Run a demonstration of the per-intersection querying functionality."""
    print("Traffic Control Environment - Per-Intersection Querying Demo")
    print("=" * 80)
    
    # Create an environment with multiple intersections
    num_intersections = 16
    env = TrafficControlEnv(num_intersections=num_intersections)
    
    # Reset the environment
    state = env.reset()
    
    # Run for a few steps
    for step in range(5):
        print(f"\nStep {step + 1}")
        print("-" * 40)
        
        # Get prompts for each intersection
        intersection_prompts = []
        for i in range(num_intersections):
         
            prompt = env.get_formatted_intersection_prompt(i)
            intersection_prompts.append(prompt)
            if i == 0:
                print(f"\nPrompt for Intersection {i + 1} (excerpt):")
                # Print just the first few lines of the prompt
                print(f"\n{prompt}")
        
        # Simulate querying an LLM for each intersection
        print("\nQuerying LLM for each intersection...")
        intersection_actions = []
        for i, prompt in enumerate(intersection_prompts):
            # In a real application, this would be an actual LLM API call
            llm_response = simulate_llm_response(prompt)
            if i == 0:
                print(f"LLM response for Intersection {i + 1}: {llm_response}")
            
            # Extract action from the response
            action = env.extract_action_for_intersection(llm_response)
            if i == 0:
                print(f"Extracted action: {action}")
            
            # Set the action for this intersection
            env.set_action_for_intersection(i, action)
            intersection_actions.append(action)
        
        # Check if all actions are collected
        if not env.are_all_actions_collected():
            print("Warning: Not all actions were collected!")
            missing = [i for i, m in enumerate(env.intersection_action_mask) if not m]
            print(f"Missing actions for intersections: {missing}")
        else:
            print("All actions collected successfully!")
        
        # Take a step with the collected actions
        print("\nTaking a step with the collected actions...")
        try:
            _, reward, done, info = env.step_with_per_intersection_actions()
            print("Step completed successfully")
        except Exception as e:
            print(f"Error during step: {e}")
            # Try to get more info about the environment state
            print(f"Environment state: {env._get_state_dict() if hasattr(env, '_get_state_dict') else 'Not available'}")
        
        print(f"Actions: {intersection_actions}")
        print(f"Reward: {reward:.2f}")
        
        if done:
            print("\nEpisode finished!")
            break
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    run_per_intersection_demo() 