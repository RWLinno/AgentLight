import os
import json
import argparse
import numpy as np
import torch
from traffic.traffic_adapter import TrafficControlEnv
from ragen.policy.llm_policy import LLMPolicy

def main(args):
    """Test trained traffic control model."""
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Update traffic file if specified
    if args.traffic_file:
        config['env_kwargs']['dic_traffic_env_conf']['TRAFFIC_FILE'] = args.traffic_file
    
    # Create environment
    env = TrafficControlEnv(config['env_kwargs'])
    
    # Load policy
    policy = LLMPolicy.from_pretrained(args.model_path)
    
    # Run episodes
    rewards = []
    waiting_vehicles = []
    
    for episode in range(args.num_episodes):
        print(f"Episode {episode+1}/{args.num_episodes}")
        
        # Reset environment
        state = env.reset(seed=args.seed + episode if args.seed is not None else None)
        
        episode_reward = 0
        episode_waiting = []
        
        for step in range(args.max_steps):
            # Get action from policy
            action = policy.predict(state)
            print(f"Step {step+1}, Action: {action}")
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_waiting.append(sum(sum(s.get("lane_num_waiting_vehicle_in", [])) for s in env.state))
            
            # Print step information
            print(f"  Reward: {reward:.2f}, Done: {done}")
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Record episode metrics
        rewards.append(episode_reward)
        waiting_vehicles.append(np.mean(episode_waiting))
        
        print(f"Episode {episode+1} finished")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Average Waiting Vehicles: {np.mean(episode_waiting):.2f}")
    
    # Print overall metrics
    print("\nTest Results:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Waiting Vehicles: {np.mean(waiting_vehicles):.2f} ± {np.std(waiting_vehicles):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test traffic control model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--config_path", type=str, default="config/traffic.yaml",
                        help="Path to environment configuration")
    parser.add_argument("--traffic_file", type=str, default=None,
                        help="Traffic flow file to use (optional)")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args) 