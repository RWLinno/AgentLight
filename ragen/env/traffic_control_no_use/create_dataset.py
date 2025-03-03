"""
Preprocess dataset for traffic control task
"""

import os
import json
from datasets import Dataset
import argparse
from verl.utils.hdfs_io import copy, makedirs
import random
import numpy as np

from ragen.env.traffic_control_no_use import TrafficControlEnv

INSTRUCTION_TEMPLATE = """You are a traffic signal control agent.

Traffic Control Quick Guide:
Goal: Optimize traffic flow by controlling signal phases at intersections.

Phases:
- Each phase (0-7) controls which traffic directions have green lights

Metrics:
- Queue Length: Number of vehicles waiting at each lane
- Waiting Time: Time vehicles have been waiting
- Traffic Pressure: Imbalance of vehicles across lanes

Actions:
Choose a phase number from 0 to 7 to control all intersections.

Rewards:
- Negative queue length: -0.25 per vehicle in queue
- Your goal is to maximize rewards by minimizing delays

[Current Traffic State]:
{observation}
Decide which phase to implement next:\
"""

templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

def generate_realistic_flow(output_path, duration=3600, peak_duration=900, seed=42):
    """
    Generate realistic traffic flow patterns with morning and evening peak hours.
    
    Args:
        output_path: Path to save the flow file
        duration: Total simulation duration in seconds (default: 1 hour)
        peak_duration: Duration of peak hours in seconds (default: 15 minutes)
        seed: Random seed for reproducibility
        
    Returns:
        Path to the generated flow file
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Define vehicle types with probabilities
    vehicle_types = {
        "sedan": 0.6,  # Regular passenger cars
        "bus": 0.05,   # Buses
        "truck": 0.1,  # Trucks
        "suv": 0.25    # SUVs
    }
    
    # Define direction probabilities for different times of day
    # Format: [north_to_south, south_to_north, east_to_west, west_to_east]
    direction_probs = {
        "morning_peak": [0.3, 0.15, 0.4, 0.15],  # Morning commute (into city)
        "evening_peak": [0.15, 0.3, 0.15, 0.4],  # Evening commute (out of city)
        "normal": [0.25, 0.25, 0.25, 0.25]       # Normal balanced traffic
    }
    
    # Define flow rate patterns (vehicles per minute)
    flow_rates = {
        "morning_peak": 30,  # High flow during morning peak
        "evening_peak": 30,  # High flow during evening peak
        "normal": 10         # Normal flow during off-peak
    }
    
    # Determine times for morning and evening peaks
    morning_peak_start = int(duration * 0.2)  # Morning peak at 20% of simulation time
    evening_peak_start = int(duration * 0.7)  # Evening peak at 70% of simulation time
    
    # Initialize flow data
    flow = []
    vehicle_id = 0
    
    # Helper function to determine flow type based on time
    def get_flow_type(time):
        if morning_peak_start <= time < morning_peak_start + peak_duration:
            return "morning_peak"
        elif evening_peak_start <= time < evening_peak_start + peak_duration:
            return "evening_peak"
        else:
            return "normal"
    
    # Helper function to generate a vehicle
    def generate_vehicle(start_time, direction_idx):
        nonlocal vehicle_id
        
        # Select vehicle type based on probabilities
        vehicle_type = random.choices(
            list(vehicle_types.keys()),
            weights=list(vehicle_types.values())
        )[0]
        
        # Define route based on direction index
        routes = [
            ["road_north_1", "road_south_1"],  # North to South
            ["road_south_1", "road_north_1"],  # South to North
            ["road_east_1", "road_west_1"],    # East to West
            ["road_west_1", "road_east_1"]     # West to East
        ]
        
        route = routes[direction_idx]
        
        # Define vehicle properties based on type
        max_speed = {
            "sedan": 16.67,  # 60 km/h
            "bus": 13.89,    # 50 km/h
            "truck": 13.89,  # 50 km/h
            "suv": 16.67     # 60 km/h
        }
        
        # Create vehicle entry
        vehicle = {
            "vehicle_id": f"veh_{vehicle_id}",
            "route": route,
            "start_time": start_time,
            "end_time": -1,  # Will be determined by simulation
            "type": vehicle_type,
            "max_speed": max_speed[vehicle_type]
        }
        
        vehicle_id += 1
        return vehicle
    
    # Generate vehicles for the entire duration
    current_time = 0
    while current_time < duration:
        flow_type = get_flow_type(current_time)
        
        # Determine how many vehicles to generate in this time step
        flow_rate = flow_rates[flow_type]
        vehicles_per_second = flow_rate / 60
        
        # Poisson distribution for realistic arrival patterns
        num_vehicles = np.random.poisson(vehicles_per_second)
        
        # Generate vehicles with directions based on time of day
        for _ in range(num_vehicles):
            direction_idx = random.choices(
                range(4),
                weights=direction_probs[flow_type]
            )[0]
            
            vehicle = generate_vehicle(current_time, direction_idx)
            flow.append(vehicle)
        
        current_time += 1  # Move to next second
    
    # Save to JSON file
    flow_file = os.path.join(output_path, "realistic_flow.json")
    with open(flow_file, 'w') as f:
        json.dump(flow, f, indent=2)
    
    print(f"Generated realistic traffic flow with {len(flow)} vehicles")
    return flow_file

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate traffic control dataset.")
    parser.add_argument("--env", type=str, default="traffic_control", help="Environment name (default: 'traffic_control').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/traffic", help="Output directory (default: 'data/traffic').")
    parser.add_argument("--train_size", type=int, default=100, help="Number of training examples (default: 100).")
    parser.add_argument("--test_size", type=int, default=10, help="Number of test examples (default: 10).")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])

    args = parser.parse_args()
    
    assert args.env == "traffic_control", f"Unsupported environment: {args.env}"
    data_source = args.env
    
    # Create environment
    env = TrafficControlEnv(
        num_intersections=4,
        action_space=8,
        min_action_time=10, 
        max_steps=100
    )
    
    # Generate data
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    
    for seed in seeds:
        observation = env.reset(seed=seed)
        instruction = INSTRUCTION_TEMPLATE.format(observation=observation)
        instructions.append(instruction)
    
    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "traffic_control",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train" if idx < args.train_size else "test", "index": idx}
        }
    
    # Create datasets
    dataset = [_create_instance(i, instructions[i]) for i in range(len(instructions))]
    train_dataset = Dataset.from_list(dataset[:args.train_size])
    test_dataset = Dataset.from_list(dataset[args.train_size:])

    # Save datasets
    os.makedirs(args.output, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))
    
    print(f"Created {args.train_size} training examples and {args.test_size} test examples in {args.output}")

    flow_path = os.path.join(args.output, "realistic_flow.json")
    generate_realistic_flow(flow_path)
    
    config_path = os.path.join(args.output, "config.json")
    config = {
        "interval": 1.0,
        "seed": 0,
        "dir": args.output,
        "roadnetFile": "roadnet_1x1.json",
        "flowFile": "realistic_flow.json",
        "rlTrafficLight": True,
        "saveReplay": False
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()