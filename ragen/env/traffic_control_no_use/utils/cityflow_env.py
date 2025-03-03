import os
import logging
import json
import numpy as np
import random
from collections import OrderedDict
from typing import Dict, List, Union, Any, Tuple

class CityFlowEnv:
    """
    CityFlow Environment for Traffic Signal Control
    """
    
    def __init__(self, path_to_log: str, path_to_work_directory: str, 
                 dic_traffic_env_conf: Dict, dic_path: Dict) -> None:
        """
        Initialize the environment
        
        Args:
            path_to_log: Path to save logs
            path_to_work_directory: Path to save configuration files
            dic_traffic_env_conf: Traffic environment configuration
            dic_path: Dictionary of paths
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        
        # Logger setup
        self.logger = logging.getLogger('cityflow_test')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            self.logger.addHandler(console_handler)
            
        # Extract important configuration settings
        self.num_intersections = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_phases = self.dic_traffic_env_conf['NUM_PHASES']
        self.num_lanes = self.dic_traffic_env_conf['NUM_LANES']
        self.yellow_time = self.dic_traffic_env_conf['YELLOW_TIME']
        self.min_action_time = self.dic_traffic_env_conf['MIN_ACTION_TIME']
        
        # Generate intersection dictionary based on grid network
        self.intersection_dict = self._build_intersection_dict()
        
        # Create necessary configuration files before initializing the engine
        self._create_config_files()
        
        import cityflow
        self.eng = cityflow.Engine(os.path.join(self.path_to_work_directory, "config.json"))
        self.logger.info("Successfully imported cityflow")
            
        # Initialize intersection objects (in real implementation these would manage individual intersections)
        self.intersections = []
        self.current_phase = {intersection_id: 0 for intersection_id in self.intersection_dict.keys()}
        
        # Record statistics for all intersections
        self.state = None
        self.current_time = 0

    def _build_intersection_dict(self) -> Dict:
        """Create a dictionary of intersection information based on grid network"""
        size = int(np.sqrt(self.num_intersections))
        intersection_dict = OrderedDict()
        
        for i in range(size):
            for j in range(size):
                intersection_id = f"intersection_{i}_{j}"
                intersection_dict[intersection_id] = {
                    "id": intersection_id,
                    "point": {"x": i, "y": j},
                    "lanes": [f"lane_{i}_{j}_{k}" for k in range(self.num_lanes)]
                }
                
        return intersection_dict
    
    def _create_config_files(self) -> None:
        """Create configuration files for CityFlow if they don't exist"""
        # Check if roadnet file exists
        roadnet_path = os.path.join(self.path_to_work_directory, "roadnet.json")
        if not os.path.exists(roadnet_path):
            # Create a simple 2x2 grid network
            self.logger.info(f"Creating roadnet file at {roadnet_path}")
            roadnet = {
                "intersections": [
                    {"id": "intersection_0_0", "point": {"x": 0, "y": 0}},
                    {"id": "intersection_0_1", "point": {"x": 0, "y": 1}},
                    {"id": "intersection_1_0", "point": {"x": 1, "y": 0}},
                    {"id": "intersection_1_1", "point": {"x": 1, "y": 1}}
                ],
                "roads": []
            }
            
            with open(roadnet_path, 'w') as f:
                json.dump(roadnet, f, indent=2)
                
        # Check if flow file exists
        flow_path = os.path.join(self.path_to_work_directory, "flow.json")
        if not os.path.exists(flow_path):
            self.logger.info(f"Creating flow file at {flow_path}")
            # 添加实际的交通流量数据，不再使用空列表
            flow_data = [
                {
                    "vehicle": {
                        "length": 5.0,
                        "width": 2.0,
                        "maxPosAcc": 2.0,
                        "maxNegAcc": 4.5,
                        "usualPosAcc": 2.0,
                        "usualNegAcc": 4.5,
                        "minGap": 2.5,
                        "maxSpeed": 16.67,
                        "headwayTime": 1.5
                    },
                    "route": ["road_0_1_0", "road_1_1_0"],
                    "interval": 5.0,
                    "startTime": 0,
                    "endTime": 3600
                },
                {
                    "vehicle": {
                        "length": 5.0,
                        "width": 2.0,
                        "maxPosAcc": 2.0,
                        "maxNegAcc": 4.5,
                        "usualPosAcc": 2.0,
                        "usualNegAcc": 4.5,
                        "minGap": 2.5,
                        "maxSpeed": 16.67,
                        "headwayTime": 1.5
                    },
                    "route": ["road_1_0_0", "road_1_1_0"],
                    "interval": 8.0,
                    "startTime": 0,
                    "endTime": 3600
                }
            ]
            with open(flow_path, 'w') as f:
                json.dump(flow_data, f, indent=2)
                
        # Create config file for CityFlow
        config_path = os.path.join(self.path_to_work_directory, "config.json")
        self.logger.info(f"Creating config file at {config_path}")
        config = {
            "interval": 1.0,  # 确保使用固定值而不是从配置中获取
            "seed": 0,
            "dir": self.path_to_work_directory,
            "roadnetFile": "roadnet.json",
            "flowFile": "flow.json",
            "rlTrafficLight": True,
            "saveReplay": False
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Loaded roadnet from {roadnet_path}")
            
    def reset(self) -> Dict:
        """Reset the simulation to initial state"""
        # Reset the traffic simulator
        if hasattr(self.eng, 'reset'):
            self.eng.reset()
            
        # Reset internal state
        self.current_time = 0
        self.current_phase = {intersection_id: 0 for intersection_id in self.intersection_dict.keys()}
        
        # Get initial traffic state
        self.state = self._update_traffic_state()
        
        return self.state
        
    def _update_traffic_state(self) -> Dict:
        """Update and return current traffic state"""
        state = {}
        
        # Get all lane vehicle counts and waiting counts at once
        lane_vehicle_counts = self.eng.get_lane_vehicle_count()
        lane_waiting_counts = self.eng.get_lane_waiting_vehicle_count()
        
        for intersection_id, intersection_info in self.intersection_dict.items():
            intersection_state = {}
            
            # Current phase information
            intersection_state["current_phase"] = int(self.current_phase[intersection_id])
            
            # Lane information
            lane_queue = []
            lane_waiting = []
            for lane in intersection_info["lanes"]:
                # Access the counts from the dictionaries
                queue = lane_vehicle_counts.get(lane, 0)  # Use get() with default 0 to handle missing lanes
                waiting = lane_waiting_counts.get(lane, 0)  # Use get() with default 0 to handle missing lanes
                
                lane_queue.append(queue)
                lane_waiting.append(waiting)
                
            intersection_state["lane_queue_length"] = lane_queue
            intersection_state["lane_waiting_vehicle_count"] = lane_waiting
            
            # Calculate pressure (imbalance of vehicles across incoming/outgoing lanes)
            traffic_pressure = max(lane_queue) - min(lane_queue) if lane_queue else 0
            intersection_state["traffic_pressure"] = traffic_pressure
            
            # Add to state dictionary
            state[intersection_id] = intersection_state
            
        return state
        
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the simulation given the action
        
        Args:
            action: Phase to set for the intersections
            
        Returns:
            next_state: Updated traffic state
            reward: Reward for the action
            done: Whether the simulation is done
            info: Additional information
        """
        # Update phases based on action
        for intersection_id in self.intersection_dict.keys():
            self.current_phase[intersection_id] = action
            
        # Simulate traffic for min_action_time steps
        for _ in range(self.min_action_time):
            self.eng.next_step()
            self.current_time += 1
            
        # Update traffic state
        next_state = self._update_traffic_state()
        self.state = next_state
        
        # Calculate reward
        reward = self._calculate_reward(next_state)
        
        # Check if simulation is done
        done = False  # In a real implementation, this would be based on simulation end condition
        
        # Additional info
        info = {"current_time": self.current_time}
        
        return next_state, reward, done, info
        
    def _calculate_reward(self, state: Dict) -> float:
        """
        Improved reward function with balanced incentives.
        Adapted from LLMTSCS's reward calculation approach.
        """
        # Initialize reward components
        reward_components = {
            'queue_length': 0,
            'waiting_time': 0,
            'throughput': 0,
            'pressure': 0,
            'phase_switching': 0
        }
        
        # Previous and current queue metrics
        prev_total_queue = 0
        curr_total_queue = 0
        
        # Get previous queue lengths
        if hasattr(self, 'previous_state') and self.previous_state:
            for _, intersection_state in self.previous_state.items():
                if 'lane_queue_length' in intersection_state:
                    prev_total_queue += sum(intersection_state['lane_queue_length'])
        
        # Get current queue lengths
        for _, intersection_state in state.items():
            if 'lane_queue_length' in intersection_state:
                curr_total_queue += sum(intersection_state['lane_queue_length'])
        
        # Calculate queue change (positive if queue reduced)
        queue_change = prev_total_queue - curr_total_queue
        
        # Apply different reward components with appropriate scaling
        # 1. Queue length penalty (lower weight to avoid excessive negative rewards)
        reward_components['queue_length'] = -0.1 * curr_total_queue
        
        # 2. Queue change reward (positive reinforcement for reducing queues)
        reward_components['queue_change'] = 0.5 * queue_change
        
        # 3. Waiting time penalty
        waiting_vehicles = 0
        for _, intersection_state in state.items():
            if 'lane_waiting_vehicle_count' in intersection_state:
                waiting_vehicles += sum(intersection_state['lane_waiting_vehicle_count'])
        reward_components['waiting_time'] = -0.1 * waiting_vehicles
        
        # 4. Throughput reward (if available)
        if hasattr(self.eng, 'get_finished_vehicle_count'):
            try:
                finished_vehicles = self.eng.get_finished_vehicle_count()
                reward_components['throughput'] = 0.5 * finished_vehicles
            except:
                pass
        
        # 5. Phase switching penalty (only if phase changed from last step)
        if hasattr(self, 'last_action') and self.last_action is not None:
            for i in range(len(self.current_phase)):
                if self.current_phase[i] != self.last_action:
                    reward_components['phase_switching'] = -0.1  # Small penalty for changing phases
        
        # Store current state and action for next comparison
        self.previous_state = state
        self.last_action = self.current_phase.copy()
        
        # Calculate final reward with better balance
        total_reward = sum(reward_components.values())
        
        # Reward scaling to avoid extremely large negative values
        total_reward = max(total_reward, -10.0)  # Clip very negative rewards
        
        # Debug information
        self.reward_components = reward_components
        
        return total_reward 

    def get_state_for_prompt(self) -> Dict:
        """
        Returns state information formatted specifically for prompt generation.
        This matches the expected format in generate_traffic_prompt()
        """
        if self.state is None:
            self.state = self._update_traffic_state()
        
        prompt_state = {
            "vehicle_counts": {},
            "waiting_counts": {},
            "current_phase": {},
            "traffic_pressure": {}
        }
        
        # Format the state data for prompt generation
        for intersection_id, intersection_info in self.state.items():
            # Add current phase
            prompt_state["current_phase"][intersection_id] = intersection_info.get("current_phase", 0)
            
            # Add traffic pressure
            prompt_state["traffic_pressure"][intersection_id] = intersection_info.get("traffic_pressure", 0)
            
            # Add lane vehicle counts and waiting counts in the expected format
            if "lane_queue_length" in intersection_info:
                prompt_state["vehicle_counts"][intersection_id] = intersection_info["lane_queue_length"]
            
            if "lane_waiting_vehicle_count" in intersection_info:
                prompt_state["waiting_counts"][intersection_id] = intersection_info["lane_waiting_vehicle_count"]
        
        return prompt_state 

    def render(self, mode='human'):
        """
        Render the environment state.
        If mode is 'human', this returns a visualization.
        If mode is 'rgb_array', this returns an RGB array.
        Otherwise, it returns the environment state for prompt generation.
        """
        if self.state is None:
            self.state = self._update_traffic_state()
        
        if mode == 'rgb_array':
            # Return a visualization as RGB array (if implemented)
            return None  # Replace with actual visualization when implemented
        
        # Use the get_state_for_prompt method to ensure consistent format
        return self.get_state_for_prompt() 