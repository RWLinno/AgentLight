#!/usr/bin/env python
"""
Simplified training script without using Ray distributed framework
"""
import os
import sys
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("Initializing training...")
    set_seed(42)
    
    # Import environment
    try:
        from ragen.env.traffic_control_no_use.env_no_use import TrafficControlEnv
    except ImportError:
        from env.traffic_control.env import TrafficControlEnv
    
    # Create environment
    env_config = {
        "path_to_log": "./log",
        "path_to_work_directory": "./data/traffic",
        "dic_traffic_env_conf": {
            "NUM_INTERSECTIONS": 1,
            "PHASE": ["WSES", "NSSS", "WLEL", "NLSL"],
            "NUM_LANES": [3, 3, 3, 3],
            "MIN_ACTION_TIME": 5,
            "YELLOW_TIME": 2,
            "DIC_REWARD_INFO": {"pressure": -0.25, "queue_length": -0.25}
        },
        "dic_path": {
            "PATH_TO_DATA": "./data/traffic"
        },
        "max_steps": 100,
        "num_intersections": 1
    }
    
    env = TrafficControlEnv(**env_config)
    print("Environment created successfully!")
    
    # Load language model
    print("Loading language model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            model = model.cuda()
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Use simple random policy as fallback
        model = None
        tokenizer = None
        print("Will use random policy for training")
    
    # Simple training loop
    print("\nStarting training...")
    num_episodes = 5
    max_steps_per_episode = 100
    
    for episode in range(num_episodes):
        print(f"\nStarting episode {episode+1}/{num_episodes}")
        obs = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            print(f"Step {step+1}/{max_steps_per_episode}")
            
            # If model is available, use it to generate actions
            if model is not None and tokenizer is not None:
                # Prepare input
                inputs = tokenizer(obs, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.7,
                        do_sample=True
                    )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Model output: {generated_text}")
                
                # Extract action from output
                try:
                    # Simply try to extract a digit from the output as the action
                    for char in generated_text:
                        if char.isdigit() and int(char) in [0, 1, 2, 3]:
                            action = int(char)
                            break
                    else:
                        # If no valid action found, choose randomly
                        action = random.choice(env.get_all_actions())
                except:
                    action = random.choice(env.get_all_actions())
            else:
                # Choose action randomly
                action = random.choice(env.get_all_actions())
            
            print(f"Selected action: {action} ({env.ACTION_LOOKUP.get(action, 'Unknown')})")
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            print(f"Reward: {reward:.4f}, Cumulative reward: {episode_reward:.4f}")
            
            # Update observation
            obs = next_obs
            
            if done:
                print(f"Environment terminated at step {step+1}")
                break
        
        print(f"Episode {episode+1} completed, total reward: {episode_reward:.4f}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
