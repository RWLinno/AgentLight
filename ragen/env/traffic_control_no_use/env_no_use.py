import gymnasium as gym
import numpy as np
import os
import json
import logging
import random
from typing import Dict, List, Union, Tuple, Any
from ragen.utils import set_seed
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.traffic_control_no_use.utils.cityflow_env import CityFlowEnv

def state2text(state: Dict, phase_names: List[str] = None) -> str:
    """
    Convert traffic state into a detailed, structured text representation for LLM processing.
    
    Args:
        state: Dictionary containing traffic state information
        phase_names: Optional list of phase names for better readability
    
    Returns:
        Detailed text description of the current traffic state
    """
    if not phase_names:
        phase_names = ["East-West Through", "North-South Through", "East-West Left", "North-South Left"]
    
    # Extract key metrics
    current_phase = state.get("current_phase", 0)
    time_in_phase = state.get("time_in_phase", 0)
    waiting_vehicles_count = state.get("waiting_vehicles_count", 0)
    
    # Get queue information by direction
    north_queue = state.get("north_queue", 0)
    south_queue = state.get("south_queue", 0)
    east_queue = state.get("east_queue", 0)
    west_queue = state.get("west_queue", 0)
    
    # Get directional pressure
    north_pressure = state.get("north_pressure", 0)
    south_pressure = state.get("south_pressure", 0)
    east_pressure = state.get("east_pressure", 0)
    west_pressure = state.get("west_pressure", 0)
    
    # Calculate highest pressure directions
    highest_pressure = max(north_pressure, south_pressure, east_pressure, west_pressure)
    highest_pressure_direction = ""
    if north_pressure == highest_pressure:
        highest_pressure_direction += "North "
    if south_pressure == highest_pressure:
        highest_pressure_direction += "South "
    if east_pressure == highest_pressure:
        highest_pressure_direction += "East "
    if west_pressure == highest_pressure:
        highest_pressure_direction += "West "
    
    # Build comprehensive text description
    text = f"Traffic Signal Control Situation:\n\n"
    text += f"Current signal phase: {phase_names[current_phase]} (Phase {current_phase})\n"
    text += f"Time in current phase: {time_in_phase} seconds\n\n"
    
    text += "Queue Lengths:\n"
    text += f"- North approach: {north_queue} vehicles\n"
    text += f"- South approach: {south_queue} vehicles\n"
    text += f"- East approach: {east_queue} vehicles\n"
    text += f"- West approach: {west_queue} vehicles\n\n"
    
    text += "Traffic Pressure (incoming - outgoing vehicles):\n"
    text += f"- North approach: {north_pressure}\n"
    text += f"- South approach: {south_pressure}\n"
    text += f"- East approach: {east_pressure}\n"
    text += f"- West approach: {west_pressure}\n\n"
    
    text += f"Total waiting vehicles: {waiting_vehicles_count}\n\n"
    
    # Decision recommendations
    text += "Traffic Analysis:\n"
    if highest_pressure_direction:
        text += f"- Highest traffic pressure detected from: {highest_pressure_direction}\n"
    
    # Phase suggestions based on pressure
    if (east_pressure + west_pressure) > (north_pressure + south_pressure):
        if current_phase != 0 and current_phase != 2:
            text += "- Consider switching to East-West phases to address traffic pressure\n"
    elif (north_pressure + south_pressure) > (east_pressure + west_pressure):
        if current_phase != 1 and current_phase != 3:
            text += "- Consider switching to North-South phases to address traffic pressure\n"
    
    text += "\nWhat signal phase would you like to set next?"
    
    return text

class TrafficControlEnv(BaseDiscreteActionEnv, gym.Env):
    """
    Enhanced Traffic Control Environment for reinforcement learning with LLMs.
    
    This environment simulates traffic flow at an intersection and allows
    control of traffic signals. It provides rich state descriptions,
    realistic rewards, and supports integration with CityFlow.
    """
    
    # Define action constants
    INVALID_ACTION = -1
    
    def __init__(self, 
                path_to_work_directory=None,
                path_to_log=None,  # Added to match the expected parameter
                roadnet_file="roadnet.json",
                flow_file="flow.json",
                config_file="config.json",
                min_action_time=15,
                max_steps=300,
                num_intersections=1,
                yellow_time=5,
                parquet_path=None,
                **kwargs):  # Accept any additional parameters
        """
        Initialize the traffic control environment.
        
        Args:
            path_to_work_directory: Directory containing traffic simulation files
            path_to_log: Directory for logging (can be the same as work_directory)
            roadnet_file: Name of the road network file
            flow_file: Name of the traffic flow file
            config_file: Name of the configuration file
            min_action_time: Minimum time (seconds) between signal changes
            max_steps: Maximum number of steps in an episode
            num_intersections: Number of intersections to control
            yellow_time: Duration of yellow phase in seconds
            parquet_path: Path to parquet dataset (if any)
            **kwargs: Additional parameters passed from the trainer
        """
        super().__init__()
        
        # Environment configuration
        self.path_to_work_directory = path_to_work_directory
        self.path_to_log = path_to_log if path_to_log else path_to_work_directory
        self.roadnet_file = roadnet_file
        self.flow_file = flow_file
        self.config_file = config_file
        self.min_action_time = min_action_time
        self.max_steps = max_steps
        self.num_intersections = num_intersections
        self.yellow_time = yellow_time
        self.parquet_path = parquet_path
        
        # Store any additional parameters
        self.kwargs = kwargs
        
        # Environment state tracking
        self.current_step = 0
        self.last_action = None
        self.last_action_time = 0
        self.current_phase = 0
        self._finished = False
        self._success = False
        self.previous_waiting_vehicles = 0
        self.previous_queue_lengths = {}
        
        # Phase configuration
        self.phase_names = ["East-West Through", "North-South Through", 
                           "East-West Left", "North-South Left"]
        self.num_phases = len(self.phase_names)
        self.action_space = gym.spaces.Discrete(self.num_phases)
        
        # Initialize CityFlow environment
        self._init_cityflow_env()
        # try:
        #     self._init_cityflow_env()
        # except Exception as e:
        #     print(f"Warning: Could not initialize CityFlow environment: {e}")
        #     print("Using mock environment for testing")
        #     self.cityflow_env = None
            
        # Track various traffic metrics
        self.metrics = {
            "episode_rewards": [],
            "queue_lengths": [],
            "waiting_times": [],
            "throughput": 0
        }
    
    def _init_cityflow_env(self):
        """Initialize the CityFlow environment"""
        if self.path_to_work_directory:
            self._create_config_files()
            
            # Define paths and configurations for CityFlow
            dic_path = {
                "PATH_TO_WORK_DIRECTORY": self.path_to_work_directory,
                "PATH_TO_DATA": self.path_to_work_directory
            }
            
            dic_traffic_env_conf = {
                "MIN_ACTION_TIME": self.min_action_time,
                "YELLOW_TIME": self.yellow_time,
                "NUM_INTERSECTIONS": self.num_intersections,
                "NUM_PHASES": self.num_phases,
                "NUM_LANES": 3,
                "PHASE": ["WSES", "NSSS", "WLEL", "NLSL"],
                "INTERVAL": 1.0,
                "LIST_STATE_FEATURE": [
                    "cur_phase",
                    "time_this_phase",
                    "lane_num_vehicle",
                    "lane_num_waiting_vehicle_in",
                    "traffic_movement_pressure_queue"
                ],
                "DIC_REWARD_INFO": {
                    "queue_length": -0.25,
                    "pressure": -0.25,
                    "waiting_time": -0.25,
                    "throughput": 0.25
                }
            }

            print(dic_traffic_env_conf)

            self.cityflow_env = CityFlowEnv(
                path_to_log=self.path_to_log,
                path_to_work_directory=self.path_to_work_directory,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path
            )
        
        else:
            self.cityflow_env = None
    
    def _create_config_files(self) -> None:
        """Create necessary configuration files for the traffic simulation"""
        if not os.path.exists(self.path_to_work_directory):
            os.makedirs(self.path_to_work_directory, exist_ok=True)
            
        # Make sure the config file exists
        config_path = os.path.join(self.path_to_work_directory, self.config_file)
        if not os.path.exists(config_path):
            self._update_config_file()
    
    def _update_config_file(self):
        """Update the CityFlow configuration file with current settings"""
        import json
        
        config = {
            "interval": 1.0,
            "seed": 42,
            "dir": self.path_to_work_directory,
            "roadnetFile": self.roadnet_file,
            "flowFile": self.flow_file,
            "rlTrafficLight": True,
            "saveReplay": True,
            "laneChange": False
        }
        
        config_path = os.path.join(self.path_to_work_directory, self.config_file)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional parameters for reset behavior
            
        Returns:
            State description as text
        """
        # Reset environment state
        self.current_step = 0
        self.last_action = None
        self.last_action_time = 0
        self.current_phase = 0
        self._finished = False
        self._success = False
        self.previous_waiting_vehicles = 0
        self.previous_queue_lengths = {"north": 0, "south": 0, "east": 0, "west": 0}
        
        # Reset metrics
        self.metrics = {
            "episode_rewards": [],
            "queue_lengths": [],
            "waiting_times": [],
            "throughput": 0
        }
        
        # Reset CityFlow if available
        state = {}
        if self.cityflow_env:
            raw_state = self.cityflow_env.reset()
            state = self._process_raw_state(raw_state)
        else:
            # Mock state for testing
            state = {
                "current_phase": self.current_phase,
                "time_in_phase": 0,
                "waiting_vehicles_count": 5,
                "north_queue": 2,
                "south_queue": 1,
                "east_queue": 3,
                "west_queue": 4,
                "north_pressure": 1,
                "south_pressure": 0,
                "east_pressure": 2,
                "west_pressure": 1
            }
        
        # Convert state to text for LLM
        return state2text(state, self.phase_names)
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Traffic signal phase to set (0-3)
            
        Returns:
            Tuple of (state, reward, done, info)
        """
        # Track step
        self.current_step += 1
        self.last_action = action
        
        done = self.current_step >= self.max_steps
        info = {"step": self.current_step}
        
        # Take action in CityFlow if available
        state = {}
        reward = 0
        if self.cityflow_env:
            raw_state, raw_reward, done, info = self.cityflow_env.step(action)
            state = self._process_raw_state(raw_state)
            reward = self._calculate_reward(state)
        else:
            # Mock state and reward for testing
            state = {
                "current_phase": action,
                "time_in_phase": 0,
                "waiting_vehicles_count": max(0, 5 - self.current_step),
                "north_queue": max(0, 2 - (1 if action == 1 or action == 3 else 0)),
                "south_queue": max(0, 1 - (1 if action == 1 or action == 3 else 0)),
                "east_queue": max(0, 3 - (1 if action == 0 or action == 2 else 0)),
                "west_queue": max(0, 4 - (1 if action == 0 or action == 2 else 0)),
                "north_pressure": max(0, 1 - (1 if action == 1 or action == 3 else 0)),
                "south_pressure": 0,
                "east_pressure": max(0, 2 - (1 if action == 0 or action == 2 else 0)),
                "west_pressure": max(0, 1 - (1 if action == 0 or action == 2 else 0))
            }
            
            # Calculate reward based on queue reductions
            current_waiting = state["waiting_vehicles_count"]
            reward = -0.1 * current_waiting
            if self.previous_waiting_vehicles > current_waiting:
                reward += 0.5  # Reward for reducing waiting vehicles
            self.previous_waiting_vehicles = current_waiting
        
        # Track metrics
        self.metrics["episode_rewards"].append(reward)
        self.metrics["queue_lengths"].append(
            state["north_queue"] + state["south_queue"] + 
            state["east_queue"] + state["west_queue"]
        )
        
        # Check if finished
        self._finished = done
        
        # Convert state to text
        state_description = state2text(state, self.phase_names)
        
        return state_description, reward, done, info
    
    def _process_raw_state(self, raw_state):
        """
        Process raw state from CityFlow into a structured state dict.
        """
        # 初始化基本状态结构
        processed_state = {
            "current_phase": self.current_phase,
            "time_in_phase": 0,
            "waiting_vehicles_count": 0,
            "north_queue": 0,
            "south_queue": 0,
            "east_queue": 0,
            "west_queue": 0,
            "north_pressure": 0,
            "south_pressure": 0,
            "east_pressure": 0,
            "west_pressure": 0
        }
        
        if raw_state:
            if self.cityflow_env and not hasattr(self.cityflow_env.eng, 'reset'):
                processed_state["waiting_vehicles_count"] = random.randint(1, 10)
                processed_state["north_queue"] = random.randint(0, 5)
                processed_state["south_queue"] = random.randint(0, 5)
                processed_state["east_queue"] = random.randint(0, 5)
                processed_state["west_queue"] = random.randint(0, 5)
                processed_state["north_pressure"] = random.randint(-2, 2)
                processed_state["south_pressure"] = random.randint(-2, 2)
                processed_state["east_pressure"] = random.randint(-2, 2)
                processed_state["west_pressure"] = random.randint(-2, 2)
            elif isinstance(raw_state, dict):
                for key, value in raw_state.items():
                    if 'queue_length' in value:
                        if 'north' in key:
                            processed_state["north_queue"] = value['queue_length']
                        elif 'south' in key:
                            processed_state["south_queue"] = value['queue_length']
                        elif 'east' in key:
                            processed_state["east_queue"] = value['queue_length']
                        elif 'west' in key:
                            processed_state["west_queue"] = value['queue_length']
                    
                    if 'traffic_pressure' in value:
                        if 'north' in key:
                            processed_state["north_pressure"] = value['traffic_pressure']
                        elif 'south' in key:
                            processed_state["south_pressure"] = value['traffic_pressure']
                        elif 'east' in key:
                            processed_state["east_pressure"] = value['traffic_pressure']
                        elif 'west' in key:
                            processed_state["west_pressure"] = value['traffic_pressure']
                    
                    if 'waiting_vehicle_count' in value:
                        processed_state["waiting_vehicles_count"] += value['waiting_vehicle_count']
        
            if processed_state["waiting_vehicles_count"] == 0:
                processed_state["waiting_vehicles_count"] = max(1, int(random.gauss(5, 2)))
                total_queue = processed_state["north_queue"] + processed_state["south_queue"] + \
                             processed_state["east_queue"] + processed_state["west_queue"]
                if total_queue == 0:
                    processed_state["north_queue"] = max(0, int(random.gauss(2, 1)))
                    processed_state["south_queue"] = max(0, int(random.gauss(2, 1)))
                    processed_state["east_queue"] = max(0, int(random.gauss(3, 1)))
                    processed_state["west_queue"] = max(0, int(random.gauss(3, 1)))
        
        return processed_state
    
    def _calculate_reward(self, state):
        """
        Calculate reward based on current state.
        
        This incorporates concepts from the queue-based and pressure-based
        reward functions in the LLMLight implementation.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # Queue length component (negative reward for queues)
        current_queue_sum = (
            state["north_queue"] + state["south_queue"] + 
            state["east_queue"] + state["west_queue"]
        )
        queue_penalty = -0.1 * current_queue_sum
        
        # Queue change component (positive for queue reduction)
        previous_queue_sum = sum(self.previous_queue_lengths.values())
        queue_change_reward = 0.5 * max(0, previous_queue_sum - current_queue_sum)
        
        # Pressure component
        pressure_sum = abs(state["north_pressure"]) + abs(state["south_pressure"]) + \
                      abs(state["east_pressure"]) + abs(state["west_pressure"])
        pressure_penalty = -0.05 * pressure_sum
        
        # Waiting vehicles component
        waiting_penalty = -0.1 * state["waiting_vehicles_count"]
        
        # Action switching penalty to discourage frequent changes
        action_switch_penalty = -0.1 if self.current_phase != state["current_phase"] else 0
        
        # Throughput reward (would be calculated from CityFlow in a real implementation)
        throughput_reward = 0.1 * state.get("throughput", 0)
        
        # Update previous values for next comparison
        self.previous_queue_lengths = {
            "north": state["north_queue"],
            "south": state["south_queue"],
            "east": state["east_queue"],
            "west": state["west_queue"]
        }
        
        # Update current phase
        self.current_phase = state.get("current_phase", self.current_phase)
        
        # Calculate total reward
        reward = queue_penalty + queue_change_reward + pressure_penalty + \
                waiting_penalty + action_switch_penalty + throughput_reward
        
        return reward
    
    def extract_action(self, text):
        """Extract action from text output by an LLM"""
        import re
        
        if "<answer>" in text and "</answer>" in text:
            # Extract content between <answer> tags
            pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Try to parse as integer
                    return int(matches[0].strip())
                except ValueError:
                    # If not an integer, return invalid action
                    return self.INVALID_ACTION
        
        # If no proper answer tag format, try to find the first number in the text
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                action = int(numbers[0])
                if 0 <= action < self.num_phases:
                    return action
            except ValueError:
                pass
        
        return self.INVALID_ACTION
    
    def finished(self):
        """Check if the environment is finished"""
        return self._finished
    
    def success(self):
        """Check if the environment task was successful"""
        return self._success
    
    def render(self, mode='human'):
        """Render the environment visualization"""
        # For now, just return a text representation
        if self.cityflow_env:
            return "CityFlow visualization not implemented yet"
        else:
            state = {
                "current_phase": self.current_phase,
                "time_in_phase": 0,
                "waiting_vehicles_count": self.previous_waiting_vehicles,
                "north_queue": self.previous_queue_lengths.get("north", 0),
                "south_queue": self.previous_queue_lengths.get("south", 0),
                "east_queue": self.previous_queue_lengths.get("east", 0),
                "west_queue": self.previous_queue_lengths.get("west", 0),
                "north_pressure": 0,
                "south_pressure": 0,
                "east_pressure": 0,
                "west_pressure": 0
            }
            return state2text(state, self.phase_names)
    
    def get_last_action(self) -> int:
        """Get the last action taken"""
        if self.last_action is None:
            return self.INVALID_ACTION
        return self.last_action
    
    def copy(self):
        """Create a copy of the environment"""
        new_env = TrafficControlEnv(
            path_to_work_directory=self.path_to_work_directory,
            path_to_log=self.path_to_log,
            roadnet_file=self.roadnet_file,
            flow_file=self.flow_file,
            config_file=self.config_file,
            min_action_time=self.min_action_time,
            max_steps=self.max_steps,
            num_intersections=self.num_intersections,
            yellow_time=self.yellow_time,
            parquet_path=self.parquet_path
        )
        new_env.last_action = self.last_action
        new_env.current_step = self.current_step
        new_env.current_phase = self.current_phase
        new_env._finished = self._finished
        new_env._success = self._success
        new_env.previous_waiting_vehicles = self.previous_waiting_vehicles
        new_env.previous_queue_lengths = self.previous_queue_lengths.copy()
        return new_env
    
    # Add support for modern gym interface
    def _copy_tracking_variables(self, other_env):
        """Copy tracking variables from another environment instance"""
        self.current_step = other_env.current_step
        self.current_phase = other_env.current_phase
        self.previous_waiting_vehicles = other_env.previous_waiting_vehicles
        self.previous_queue_lengths = other_env.previous_queue_lengths.copy()

if __name__ == "__main__":
    env = TrafficControlEnv()
    print(env.reset())
