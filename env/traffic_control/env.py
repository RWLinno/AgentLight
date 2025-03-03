import gym
import numpy as np
import re
import copy
import os
import json
import logging
from typing import Optional, List, Tuple, Any, Dict, Union
from typing_extensions import override

from ragen.env.base import BaseDiscreteActionEnv
from .cityflow_adapter import CityFlowAdapter

class TrafficControlEnv(BaseDiscreteActionEnv):
    """
    Traffic Control Environment for traffic signal control.
    
    ## Description
    This environment simulates a traffic network with multiple intersections.
    The agent controls the traffic signals to optimize traffic flow.
    
    ## Action Space
    For single intersection:
    The action shape is `(1,)` in the range `{1, 2, 3, 4}` indicating
    which phase to set for the intersection:
    - 0: Invalid action
    - 1: Phase 1 (WSES - West-East Straight, East-South)
    - 2: Phase 2 (NSSS - North-South Straight, South-South)
    - 3: Phase 3 (WLEL - West-Left, East-Left)
    - 4: Phase 4 (NLSL - North-Left, South-Left)
    
    For multiple intersections:
    The action shape is `(num_intersections,)` where each element is
    in the range `{1, 2, 3, 4}` indicating which phase to set for each intersection.
    
    ## Observation
    Text-based observation describing the current state of each intersection:
    - Number of vehicles on each lane
    - Waiting vehicles
    - Current signal phase
    - Traffic pressure
    
    ## Rewards
    - Negative reward for traffic pressure (-0.25 * pressure)
    - Negative reward for queue length (-0.25 * queue_length)
    - Positive reward for throughput (0.5 * throughput)
    - Penalty for invalid actions (-1.0)
    """
    
    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1.0
    
    # Define lookup for action names
    ACTION_LOOKUP = {
        0: "Invalid",
        1: "WSES (West-East Straight)",
        2: "NSSS (North-South Straight)",
        3: "WLEL (West-Left East-Left)",
        4: "NLSL (North-Left South-Left)",
    }
    
    # Define phase explanations for better understanding
    PHASE_EXPLANATIONS = {
        1: "WSES: Green light for vehicles going straight from West to East and East to West",
        2: "NSSS: Green light for vehicles going straight from North to South and South to North",
        3: "WLEL: Green light for vehicles turning left from West and East",
        4: "NLSL: Green light for vehicles turning left from North and South",
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the Traffic Control Environment.
        
        Args:
            path_to_log: Path to log directory (default: './log')
            path_to_work_directory: Path to work directory (default: './data/traffic')
            dic_traffic_env_conf: Traffic environment configuration dictionary
            dic_path: Dictionary containing paths
            max_steps: Maximum number of steps (default: 3600)
            num_intersections: Number of intersections (default: 1)
        """
        BaseDiscreteActionEnv.__init__(self)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('TrafficControlEnv')
        
        # Extract parameters with defaults
        self.path_to_log = kwargs.get('path_to_log', './log')
        self.path_to_work_directory = kwargs.get('path_to_work_directory', './data/traffic')
        self.dic_traffic_env_conf = kwargs.get('dic_traffic_env_conf', {})
        self.dic_path = kwargs.get('dic_path', {})
        self.max_steps = kwargs.get('max_steps', 3600)
        self.num_intersections = kwargs.get('num_intersections', 1)
        
        # Ensure directories exist
        os.makedirs(self.path_to_log, exist_ok=True)
        os.makedirs(self.path_to_work_directory, exist_ok=True)
        
        # Initialize default configuration if not provided
        if not self.dic_traffic_env_conf:
            self.dic_traffic_env_conf = {
                'NUM_INTERSECTIONS': self.num_intersections,
                'NUM_PHASES': 4,
                'PHASE': ["WSES", "NSSS", "WLEL", "NLSL"],
                'NUM_LANES': [3, 3, 3, 3],
                'YELLOW_TIME': 3,
                'MIN_ACTION_TIME': 10,
                'ROADNET_FILE': 'roadnet_4_4.json',
                'TRAFFIC_FILE': 'anon_4_4_hangzhou_real_5816.json',
                'DIC_REWARD_INFO': {
                    'pressure': -0.25,
                    'queue_length': -0.25,
                    'throughput': 0.5
                }
            }
        
        # Initialize CityFlow adapter
        self.adapter = CityFlowAdapter(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.path_to_work_directory,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        
        # Set up action space (discrete with 4 actions, starting from 1)
        self.num_phases = len(self.dic_traffic_env_conf.get('PHASE', ["WSES", "NSSS", "WLEL", "NLSL"]))
        
        # Create action space based on number of intersections
        if self.num_intersections == 1:
            self.ACTION_SPACE = gym.spaces.discrete.Discrete(self.num_phases)
            # Track the offset separately for action handling
            self.action_offset = 1  # To convert between 0-indexed and 1-indexed
        else:
            # For multiple intersections, we use a MultiDiscrete space
            # Fix: MultiDiscrete doesn't accept 'start' parameter in newer gym versions
            self.ACTION_SPACE = gym.spaces.multi_discrete.MultiDiscrete(
                [self.num_phases] * self.num_intersections
            )
            # Track the offset separately for action handling
            self.action_offset = 1  # To convert between 0-indexed and 1-indexed
        
        # Add support for per-intersection action collection
        self.collected_actions = [self.INVALID_ACTION] * self.num_intersections
        self.intersection_action_mask = [False] * self.num_intersections
        
        # Initialize tracking variables
        self.reward = 0
        self.step_count = 0
        self.current_state = None
        self.previous_actions = []
        self.cumulative_rewards = []
        self.last_info = None
        self._reset_tracking_variables()

        self.logger.info(self.dic_traffic_env_conf)
        
        self.logger.info(f"TrafficControlEnv initialized with {self.num_intersections} intersections")
    
    def extract_action(self, text: str) -> Union[int, List[int]]:
        """
        Extract action from text. Parse different formats of specifying phases.
        
        Args:
            text: Text containing the action
            
        Returns:
            int or list: Action code(s) (1-4 for valid phases, 0 for invalid)
                         For multiple intersections, returns a list of actions
        """
        # For multiple intersections, check for patterns like "Intersection 1: Phase 2, Intersection 2: Phase 3"
        if self.num_intersections > 1:
            actions = [self.INVALID_ACTION] * self.num_intersections
            
            # Look for intersection-specific actions
            for i in range(self.num_intersections):
                pattern = r'(?:intersection|int)(?:\s*)[#]?\s*' + str(i+1) + r'(?:\s*):(?:\s*)(?:phase|signal)(?:\s*:)?\s*([1-4])'
                match = re.search(pattern, text.lower())
                
                if match:
                    actions[i] = int(match.group(1))
                else:
                    # Alternative: Check for simpler patterns
                    simple_pattern = r'(?:int|intersection)\s*' + str(i+1) + r'\s*[\s:]\s*([1-4])'
                    match = re.search(simple_pattern, text.lower())
                    if match:
                        actions[i] = int(match.group(1))
            
            # If we didn't find intersection-specific actions, check if there's a single action for all
            if all(a == self.INVALID_ACTION for a in actions):
                # Look for patterns like "Phase: 1", "Phase 1", etc.
                pattern = r'(?:phase|signal)(?:\s*:)?\s*([1-4])'
                match = re.search(pattern, text.lower())
                
                if match:
                    action = int(match.group(1))
                    return [action] * self.num_intersections
                
                # Alternative: Check for phase names or abbreviations
                phase_patterns = {
                    r'\bwses\b|\bwest.?east\s+straight\b': 1,
                    r'\bnsss\b|\bnorth.?south\s+straight\b': 2,
                    r'\bwlel\b|\bwest.?east\s+left\b': 3,
                    r'\bnlsl\b|\bnorth.?south\s+left\b': 4
                }
                
                for pattern, action in phase_patterns.items():
                    if re.search(pattern, text.lower()):
                        return [action] * self.num_intersections
                
                # Final fallback: Check for direct numbers
                number_match = re.search(r'\b([1-4])\b', text)
                if number_match:
                    action = int(number_match.group(1))
                    return [action] * self.num_intersections
                
                # No valid action found
                return [self.INVALID_ACTION] * self.num_intersections
            
            return actions
        else:
            # Single intersection case - original implementation
            # Look for patterns like "Phase: 1", "Phase 1", "I choose signal phase: 1", etc.
            pattern = r'(?:phase|signal)(?:\s*:)?\s*([1-4])'
            match = re.search(pattern, text.lower())
            
            if match:
                return int(match.group(1))
            
            # Alternative: Check for phase names or abbreviations
            phase_patterns = {
                r'\bwses\b|\bwest.?east\s+straight\b': 1,
                r'\bnsss\b|\bnorth.?south\s+straight\b': 2,
                r'\bwlel\b|\bwest.?east\s+left\b': 3,
                r'\bnlsl\b|\bnorth.?south\s+left\b': 4
            }
            
            for pattern, action in phase_patterns.items():
                if re.search(pattern, text.lower()):
                    return action
            
            # Final fallback: Check for direct numbers
            number_match = re.search(r'\b([1-4])\b', text)
            if number_match:
                return int(number_match.group(1))
            
            # No valid action found
            return self.INVALID_ACTION
    
    def extract_action_for_intersection(self, text: str) -> int:
        """
        Extract a single action from text for a specific intersection.
        
        Args:
            text: Text containing the action
            
        Returns:
            int: Action code (1-4 for valid phases, 0 for invalid)
        """
        # Look for patterns like "Phase: 1", "Phase 1", "I choose signal phase: 1", etc.
        pattern = r'(?:phase|signal)(?:\s*:)?\s*([1-4])'
        match = re.search(pattern, text.lower())
        
        if match:
            return int(match.group(1))
        
        # Alternative: Check for phase names or abbreviations
        phase_patterns = {
            r'\bwses\b|\bwest.?east\s+straight\b': 1,
            r'\bnsss\b|\bnorth.?south\s+straight\b': 2,
            r'\bwlel\b|\bwest.?east\s+left\b': 3,
            r'\bnlsl\b|\bnorth.?south\s+left\b': 4
        }
        
        for pattern, action in phase_patterns.items():
            if re.search(pattern, text.lower()):
                return action
        
        # Final fallback: Check for direct numbers
        number_match = re.search(r'\b([1-4])\b', text)
        if number_match:
            return int(number_match.group(1))
        
        # No valid action found
        return self.INVALID_ACTION
    
    def reset(self, seed=None, mode='text'):
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Random seed (default: None)
            mode: Rendering mode (default: 'text')
            
        Returns:
            observation: Text-based representation of the environment state
        """
        # Reset the CityFlow adapter
        state_list = self.adapter.reset()
        
        # Reset tracking variables
        self.reward = 0
        self.step_count = 0
        self.previous_actions = []
        self.cumulative_rewards = []
        self._reset_tracking_variables()
        
        # Process the adapter's state to generate text observation
        print(f"Initial state type: {type(state_list)}")
        print(f"Initial state content: {state_list[:100] if isinstance(state_list, list) else 'Not a list'}")
        
        # Store the raw state
        self.current_state = state_list
        
        # Generate a text representation for return
        if mode == 'text':
            observation = self.render(mode='text')
            return observation
        else:
            return state_list
    
    def get_intersection_observation(self, intersection_id: int, mode='text') -> str:
        """
        Get observation for a specific intersection.
        
        Args:
            intersection_id: ID of the intersection (0-based index)
            mode: Rendering mode ('text' or 'rgb_array')
            
        Returns:
            str: Text-based observation for the specified intersection
        """
        # First, check if current_state exists
        if not hasattr(self, 'current_state') or self.current_state is None:
            print(f"current_state: {self.current_state}")
            return f"No state information available for Intersection {intersection_id + 1}."
        
        # Handle different current_state structures - it could be a list or a dict
        try:
            # Handle case where current_state is a list
            if isinstance(self.current_state, list):
                if intersection_id < 0 or intersection_id >= len(self.current_state):
                    print(f"Intersection ID {intersection_id} is out of range. Available: 0-{len(self.current_state)-1}")
                    return f"Intersection ID {intersection_id + 1} is out of range."
                
                intersection_state = self.current_state[intersection_id]
            # Handle case where current_state is a dict with integer keys
            elif isinstance(self.current_state, dict) and intersection_id in self.current_state:
                intersection_state = self.current_state[intersection_id]
            # Handle case where current_state is a dict with string keys
            elif isinstance(self.current_state, dict) and str(intersection_id) in self.current_state:
                intersection_state = self.current_state[str(intersection_id)]
            else:
                print(f"No state information available for Intersection {intersection_id + 1}.")
                return f"No state information available for Intersection {intersection_id + 1}."
        except Exception as e:
            print(f"Error accessing state for Intersection {intersection_id + 1}: {str(e)}")
            return f"Error accessing state for Intersection {intersection_id + 1}: {str(e)}"
        
        # Build a text representation of the traffic state for this intersection
        text = "Traffic State Summary:\n"
        text += f"Step: {self.step_count}/{self.max_steps}\n\n"
        
        text += f"Intersection {intersection_id + 1}:\n"
        text += "------------------------\n"
        
        # Current phase
        if isinstance(intersection_state, dict) and 'current_phase' in intersection_state:
            phase_idx = intersection_state['current_phase']
            phase_name = self.ACTION_LOOKUP.get(phase_idx + 1, "Unknown")
            text += f"Current Phase: {phase_name}\n"
        
        # Traffic pressure
        if isinstance(intersection_state, dict) and 'pressure' in intersection_state:
            text += f"Traffic Pressure: {intersection_state['pressure']:.2f}\n"
        
        # Vehicle counts by approach direction
        if isinstance(intersection_state, dict) and 'lane_vehicle_count' in intersection_state:
            counts = intersection_state['lane_vehicle_count']
            directions = ["North", "East", "South", "West"]
            total_lanes = len(counts)
            lanes_per_direction = total_lanes // 4 if total_lanes >= 4 else 1
            
            text += "\nVehicle Counts by Direction:\n"
            for i, direction in enumerate(directions):
                if i * lanes_per_direction < total_lanes:
                    start_idx = i * lanes_per_direction
                    end_idx = min(start_idx + lanes_per_direction, total_lanes)
                    direction_counts = counts[start_idx:end_idx]
                    total = sum(direction_counts)
                    text += f"  {direction}: {total} vehicles\n"
        
        # Waiting vehicles
        if isinstance(intersection_state, dict) and 'lane_waiting_vehicle_count' in intersection_state:
            waiting_counts = intersection_state['lane_waiting_vehicle_count']
            
            text += "\nWaiting Vehicles by Direction:\n"
            for i, direction in enumerate(directions):
                if i * lanes_per_direction < len(waiting_counts):
                    start_idx = i * lanes_per_direction
                    end_idx = min(start_idx + lanes_per_direction, len(waiting_counts))
                    direction_waiting = waiting_counts[start_idx:end_idx]
                    total_waiting = sum(direction_waiting)
                    text += f"  {direction}: {total_waiting} waiting\n"
        
        # Lane details
        text += "\nDetailed Lane Information:\n"
        if isinstance(intersection_state, dict) and 'lane_vehicle_count' in intersection_state and 'lane_waiting_vehicle_count' in intersection_state:
            counts = intersection_state['lane_vehicle_count']
            waiting = intersection_state['lane_waiting_vehicle_count']
            
            for i, (count, wait) in enumerate(zip(counts, waiting)):
                if i < total_lanes:
                    direction = directions[i // lanes_per_direction] if i // lanes_per_direction < len(directions) else "Other"
                    lane = i % lanes_per_direction + 1
                    text += f"  {direction} Lane {lane}: {count} vehicles ({wait} waiting)\n"
        
        # Add action history if available
        if hasattr(self, 'previous_actions') and len(self.previous_actions) > 0:
            text += "\nRecent Actions for this Intersection:\n"
            for i, a in enumerate(self.previous_actions[-5:]):
                step_num = self.step_count - len(self.previous_actions[-5:]) + i
                if isinstance(a, list) and len(a) > intersection_id:
                    action_name = self.ACTION_LOOKUP.get(a[intersection_id], "Unknown")
                    text += f"Step {step_num}: {action_name}\n"
                elif not isinstance(a, list) and intersection_id == 0:
                    action_name = self.ACTION_LOOKUP.get(a, "Unknown")
                    text += f"Step {step_num}: {action_name}\n"
        
        # Add prompt for this intersection
        text += f"\n## Control Task for Intersection {intersection_id + 1}:\n"
        text += "Analyze the traffic conditions and select the optimal signal phase (1-4) for this intersection.\n"
        text += "Respond with just the phase number or name.\n"
        
        return text
    
    def get_formatted_intersection_prompt(self, intersection_id: int) -> str:
        """
        Get a formatted prompt for a specific intersection, suitable for LLM querying.
        
        Args:
            intersection_id: ID of the intersection (0-based index)
            
        Returns:
            str: Formatted prompt for the specified intersection
        """
        try:
            # Get the basic observation for this intersection
        
            observation = self.get_intersection_observation(intersection_id)
            #print(f"Got observation of length: {len(observation)}")
            
            # Create a complete prompt with instructions
            prompt = (
                "# Traffic Signal Control Task\n\n"
                "You are an intelligent traffic signal controller. Your task is to optimize "
                "traffic flow at Intersection by selecting the appropriate signal phase.\n\n"
                " Current Traffic State:\n\n" + observation + "\n\n"
                " Available Signal Phases:\n"
                "1. WSES: Green for West-East Straight traffic\n"
                "2. NSSS: Green for North-South Straight traffic\n"
                "3. WLEL: Green for West-East Left-turn traffic\n"
                "4. NLSL: Green for North-South Left-turn traffic\n\n"
                f"Based on your analysis, select the optimal signal phase (1-4) for Intersection {intersection_id + 1}."
            )
            
            return prompt
        except Exception as e:
            print(f"Error generating prompt for intersection {intersection_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a safe default prompt
            return (
                "# Traffic Signal Control Task\n\n"
                "You are an intelligent traffic signal controller. Your task is to optimize "
                "traffic flow at Intersection by selecting the appropriate signal phase.\n\n"
                " Current Traffic State:\n\n[Error retrieving traffic state]\n\n"
                " Available Signal Phases:\n"
                "1. WSES: Green for West-East Straight traffic\n"
                "2. NSSS: Green for North-South Straight traffic\n"
                "3. WLEL: Green for West-East Left-turn traffic\n"
                "4. NLSL: Green for North-South Left-turn traffic\n\n"
                f"Based on your analysis, select the optimal signal phase (1-4) for Intersection {intersection_id + 1}."
            )

    def set_action_for_intersection(self, intersection_id: int, action: int) -> bool:
        """
        Set an action for a specific intersection.
        
        Args:
            intersection_id: ID of the intersection (0-based index)
            action: Action to set (1-4 for valid phases, 0 for invalid)
            
        Returns:
            bool: True if the action is valid and successfully set, False otherwise
        """
        # Validate intersection ID
        if intersection_id < 0 or intersection_id >= self.num_intersections:
            self.logger.warning(f"Invalid intersection ID: {intersection_id}")
            return False
        
        # Validate action
        if action not in range(1, self.num_phases + 1):
            self.logger.warning(f"Invalid action {action} for intersection {intersection_id}")
            self.collected_actions[intersection_id] = self.INVALID_ACTION
            self.intersection_action_mask[intersection_id] = True
            return False
        
        # Set the action
        self.collected_actions[intersection_id] = action
        self.intersection_action_mask[intersection_id] = True
        return True
    
    def reset_collected_actions(self):
        """Reset collected actions for all intersections."""
        self.collected_actions = [self.INVALID_ACTION] * self.num_intersections
        self.intersection_action_mask = [False] * self.num_intersections
    
    def are_all_actions_collected(self) -> bool:
        """
        Check if actions have been collected for all intersections.
        
        Returns:
            bool: True if all intersections have actions set, False otherwise
        """
        return all(self.intersection_action_mask)
    
    def get_collected_actions(self) -> List[int]:
        """
        Get the currently collected actions.
        
        Returns:
            List[int]: List of actions for each intersection
        """
        return self.collected_actions.copy()
    
    def step(self, action: Union[int, List[int]]) -> Tuple[Any, float, bool, Dict]:
        """
        Step the environment forward based on an action.
        
        Args:
            action: Action to take (single int or list of ints)
            
        Returns:
            observation: Text-based observation of the environment
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        try:
            print(f"Step method called with action: {action}")
            
            # Handle multidiscrete space (range 0-3) vs. 1-indexed actions (range 1-4)
            if self.num_intersections > 1:
                # For multiple intersections
                if isinstance(action, list) and len(action) == self.num_intersections:
                    print(f"Processing multiple intersection actions: {action}")
                    # Ensure all actions are within valid range
                    for i, a in enumerate(action):
                        if a < 1 or a > 4:
                            print(f"Warning: Action {a} for intersection {i} is out of valid range (1-4). Setting to 1.")
                            action[i] = 1
                else:
                    error_msg = f"Invalid action format for multi-intersection environment: {action}"
                    print(error_msg)
                    return self.render(), 0, False, {"error": error_msg}
            else:
                # For single intersection
                if isinstance(action, int):
                    if action < 1 or action > 4:
                        print(f"Warning: Action {action} is out of valid range (1-4). Setting to 1.")
                        action = 1
                else:
                    error_msg = f"Invalid action format for single-intersection environment: {action}"
                    print(error_msg)
                    return self.render(), 0, False, {"error": error_msg}
                    
            # Take a step in the CityFlow adapter
            print("Taking step in adapter...")
            next_state, reward, done, info = self.adapter.step(action)
            print(f"Adapter step complete. Reward: {reward}, Done: {done}")
            print(f"Next state type: {type(next_state)}")
            print(f"Next state content: {next_state[:100] if isinstance(next_state, list) else 'Not a list'}")
            
            # Update tracking variables
            self.previous_actions.append(action)
            self.step_count += 1
            self.reward += reward
            self.cumulative_rewards.append(reward)
            self.current_state = next_state  # Store the raw state
            self.last_info = info
            
            # Convert the state to text observation if needed
            observation = self.render()
            enhanced_observation = self._enhance_observation(observation, action, reward)
            
            return enhanced_observation, reward, done, info
            
        except Exception as e:
            error_msg = f"Error in step method: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return self.render(), 0, False, {"error": error_msg}
    
    def step_with_per_intersection_actions(self) -> Tuple[Any, float, bool, Dict]:
        """
        Take a step using the previously collected per-intersection actions.
        
        Returns:
            observation: Text-based observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Check if all actions have been collected
        if not self.are_all_actions_collected():
            missing = [i for i, m in enumerate(self.intersection_action_mask) if not m]
            self.logger.warning(f"Missing actions for intersections: {missing}")
            return None, 0, False, {"error": f"Missing actions for intersections: {missing}"}
        
        # Get the collected actions
        actions = self.get_collected_actions()
        print(f"Actions: {actions}")
        
        # Take a step with the collected actions
        return self.step(actions)
    
    def _enhance_observation(self, observation: str, action: Union[int, List[int]], reward: float) -> str:
        """
        Enhance the observation with additional useful information.
        
        Args:
            observation: Raw observation string
            action: Last action taken
            reward: Reward received
            
        Returns:
            str: Enhanced observation
        """
        # Add action history
        action_history = "## Recent Actions:\n"
        for i, a in enumerate(self.previous_actions[-5:]):
            step_num = self.step_count - len(self.previous_actions[-5:]) + i
            
            if self.num_intersections == 1:
                action_name = self.ACTION_LOOKUP.get(a, "Unknown")
                action_history += f"Step {step_num}: {action_name}\n"
            else:
                # For multiple intersections
                action_history += f"Step {step_num}:\n"
                for int_idx, int_action in enumerate(a):
                    action_name = self.ACTION_LOOKUP.get(int_action, "Unknown")
                    action_history += f"  Intersection {int_idx + 1}: {action_name}\n"
        
        # Add reward information
        reward_info = (
            f"## Reward Information:\n"
            f"Last Reward: {reward:.2f}\n"
            f"Cumulative Reward: {self.reward:.2f}\n"
        )
        
        # Add guidance for reasoning
        reasoning_guide = (
            f"## Analysis Guidance:\n"
            f"1. Consider the number of vehicles in each direction\n"
            f"2. Pay attention to waiting vehicles (stopped at intersections)\n"
            f"3. Try to reduce traffic pressure and queue length\n"
            f"4. Check which directions have the most congestion\n"
            f"5. Consider the impact of your decision on future traffic flow\n\n"
        )
        
        # Add action prompt based on number of intersections
        if self.num_intersections > 1:
            reasoning_guide += (
                f"Based on your analysis, select the optimal signal phase (1-4) for each intersection, "
                f"in the format:\n"
                f"Intersection 1: Phase X\n"
                f"Intersection 2: Phase Y\n..."
            )
        else:
            reasoning_guide += f"Based on your analysis, select the optimal signal phase (1-4)."
        
        return f"{observation}\n\n{action_history}\n{reward_info}\n{reasoning_guide}"
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic metrics.
        
        Returns:
            float: Reward value
        """
        if not self.current_state:
            return 0.0
        
        # Extract reward components from config
        reward_info = self.dic_traffic_env_conf.get('DIC_REWARD_INFO', {
            'pressure': -0.25,
            'queue_length': -0.25,
            'throughput': 0.5
        })
        
        # Initialize metrics
        total_pressure = 0
        total_queue_length = 0
        total_throughput = 0
        
        # Calculate metrics for each intersection
        for intersection_id in range(self.num_intersections):
            if intersection_id in self.current_state:
                intersection_state = self.current_state[intersection_id]
                
                # Traffic pressure (difference between incoming and outgoing vehicles)
                if 'pressure' in intersection_state:
                    total_pressure += abs(intersection_state['pressure'])
                
                # Queue length (vehicles waiting at the intersection)
                if 'lane_queue' in intersection_state:
                    total_queue_length += sum(intersection_state['lane_queue'])
                elif 'lane_waiting_vehicle_count' in intersection_state:
                    total_queue_length += sum(intersection_state['lane_waiting_vehicle_count'])
                
                # Throughput (vehicles that successfully passed through)
                if 'throughput' in intersection_state:
                    total_throughput += intersection_state['throughput']
                # Estimate throughput if not directly available
                elif 'lane_vehicle_count' in intersection_state and len(self.previous_actions) > 0:
                    current_count = sum(intersection_state['lane_vehicle_count'])
                    # Check if we have previous state to calculate how many vehicles passed
                    if hasattr(self, '_prev_vehicle_count') and intersection_id in self._prev_vehicle_count:
                        # Vehicles that entered minus vehicles that exited
                        throughput = max(0, self._prev_vehicle_count[intersection_id] - current_count + 2)  # Adding constant for vehicles that entered
                        total_throughput += throughput
                    
                    # Store current count for next calculation
                    if not hasattr(self, '_prev_vehicle_count'):
                        self._prev_vehicle_count = {}
                    self._prev_vehicle_count[intersection_id] = current_count
        
        # Calculate reward components
        pressure_reward = reward_info.get('pressure', -0.25) * total_pressure
        queue_reward = reward_info.get('queue_length', -0.25) * total_queue_length
        throughput_reward = reward_info.get('throughput', 0.5) * total_throughput
        
        # Combine rewards
        total_reward = pressure_reward + queue_reward + throughput_reward
        
        # Log detailed reward breakdown
        self.logger.debug(f"Reward breakdown - Pressure: {pressure_reward:.2f}, "
                        f"Queue: {queue_reward:.2f}, Throughput: {throughput_reward:.2f}, "
                        f"Total: {total_reward:.2f}")
        
        return total_reward
    
    def finished(self) -> bool:
        """
        Check if the episode is finished.
        
        Returns:
            bool: True if the episode is finished, False otherwise
        """
        return self.step_count >= self.max_steps
    
    def success(self) -> bool:
        """
        Check if the episode was successful.
        In traffic control, success is measured by maintaining good traffic flow.
        
        Returns:
            bool: True if the episode was successful, False otherwise
        """
        # Define success as having an average reward above a threshold
        if not self.cumulative_rewards:
            return False
        
        avg_reward = sum(self.cumulative_rewards) / len(self.cumulative_rewards)
        success_threshold = -5.0  # This threshold can be adjusted
        
        return avg_reward > success_threshold
    
    def render(self, mode='text') -> str:
        """
        Render the environment as a text representation.
        
        Args:
            mode: Rendering mode ('text' or 'rgb_array')
            
        Returns:
            str: Text representation of the environment state
        """
        if not self.current_state:
            return "No state information available."
        
        if mode == 'rgb_array':
            # This would return an image representation if implemented
            # For now, we'll just return the text representation
            return self.render(mode='text')
        
        # Build a text representation of the traffic state
        text = "Traffic State Summary:\n"
        text += f"Step: {self.step_count}/{self.max_steps}\n\n"
        
        # Process each intersection
        for intersection_id in range(self.num_intersections):
            if intersection_id in self.current_state:
                intersection_state = self.current_state[intersection_id]
                
                text += f"Intersection {intersection_id + 1}:\n"
                text += "------------------------\n"
                
                # Current phase
                if 'current_phase' in intersection_state:
                    phase_idx = intersection_state['current_phase']
                    phase_name = self.ACTION_LOOKUP.get(phase_idx + 1, "Unknown")
                    text += f"Current Phase: {phase_name}\n"
                
                # Traffic pressure
                if 'pressure' in intersection_state:
                    text += f"Traffic Pressure: {intersection_state['pressure']:.2f}\n"
                
                # Vehicle counts by approach direction
                if 'lane_vehicle_count' in intersection_state:
                    counts = intersection_state['lane_vehicle_count']
                    directions = ["North", "East", "South", "West"]
                    total_lanes = len(counts)
                    lanes_per_direction = total_lanes // 4 if total_lanes >= 4 else 1
                    
                    text += "\nVehicle Counts by Direction:\n"
                    for i, direction in enumerate(directions):
                        if i * lanes_per_direction < total_lanes:
                            start_idx = i * lanes_per_direction
                            end_idx = min(start_idx + lanes_per_direction, total_lanes)
                            direction_counts = counts[start_idx:end_idx]
                            total = sum(direction_counts)
                            text += f"  {direction}: {total} vehicles\n"
                
                # Waiting vehicles
                if 'lane_waiting_vehicle_count' in intersection_state:
                    waiting_counts = intersection_state['lane_waiting_vehicle_count']
                    
                    text += "\nWaiting Vehicles by Direction:\n"
                    for i, direction in enumerate(directions):
                        if i * lanes_per_direction < len(waiting_counts):
                            start_idx = i * lanes_per_direction
                            end_idx = min(start_idx + lanes_per_direction, len(waiting_counts))
                            direction_waiting = waiting_counts[start_idx:end_idx]
                            total_waiting = sum(direction_waiting)
                            text += f"  {direction}: {total_waiting} waiting\n"
                
                # Lane details
                text += "\nDetailed Lane Information:\n"
                if 'lane_vehicle_count' in intersection_state and 'lane_waiting_vehicle_count' in intersection_state:
                    counts = intersection_state['lane_vehicle_count']
                    waiting = intersection_state['lane_waiting_vehicle_count']
                    
                    for i, (count, wait) in enumerate(zip(counts, waiting)):
                        if i < total_lanes:
                            direction = directions[i // lanes_per_direction] if i // lanes_per_direction < len(directions) else "Other"
                            lane = i % lanes_per_direction + 1
                            text += f"  {direction} Lane {lane}: {count} vehicles ({wait} waiting)\n"
                
                text += "\n"
        
        return text
    
    def copy(self) -> 'TrafficControlEnv':
        """
        Create a deep copy of the environment.
        
        Returns:
            TrafficControlEnv: A deep copy of the environment
        """
        new_env = TrafficControlEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.path_to_work_directory,
            dic_traffic_env_conf=copy.deepcopy(self.dic_traffic_env_conf),
            dic_path=copy.deepcopy(self.dic_path),
            max_steps=self.max_steps,
            num_intersections=self.num_intersections
        )
        
        # Copy current state
        if self.current_state is not None:
            new_env.current_state = copy.deepcopy(self.current_state)
        
        # Copy tracking variables
        new_env.step_count = self.step_count
        new_env.previous_actions = copy.deepcopy(self.previous_actions)
        new_env.cumulative_rewards = copy.deepcopy(self.cumulative_rewards)
        new_env._copy_tracking_variables(self)
        
        if hasattr(self, '_prev_vehicle_count'):
            new_env._prev_vehicle_count = self._prev_vehicle_count
        
        return new_env
    
    def get_all_actions(self) -> Union[List[int], List[List[int]]]:
        """
        Get all valid actions.
        
        Returns:
            List: List of valid actions
                  For single intersection: List of valid actions (1-4)
                  For multiple intersections: List of valid action combinations
        """
        if self.num_intersections == 1:
            return list(range(1, self.num_phases + 1))
        else:
            # For multiple intersections, this would be the cartesian product of all possible actions
            # But that's exponential, so we'll just return the valid range for each intersection
            return [list(range(1, self.num_phases + 1)) for _ in range(self.num_intersections)]
    
    def _reset_tracking_variables(self):
        """Reset tracking variables."""
        super()._reset_tracking_variables()
        self.step_count = 0
        self.previous_actions = []
        self.cumulative_rewards = []
        self._prev_vehicle_count = {}  # Initialize as a dictionary instead of a scalar
    
    def compute_score(self, solution_str: str, ground_truth: str) -> float:
        """
        Compute the score for the traffic control task.
        
        Args:
            solution_str: The solution string from the model
        """
        pass
    
    def _get_state_dict(self) -> Dict:
        """
        Get a dictionary representation of the environment state,
        suitable for use by RL agents.
        
        Returns:
            Dict: State dictionary with vehicle counts, waiting vehicles, etc.
        """
        if not self.current_state:
            return {"error": "No state information available"}
        
        state_dict = {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "total_reward": self.reward
        }
        
        # Process each intersection
        intersections = {}
        for intersection_id in range(self.num_intersections):
            if intersection_id in self.current_state:
                intersection_state = self.current_state[intersection_id]
                intersection_data = {}
                
                # Current phase
                if 'current_phase' in intersection_state:
                    phase_idx = intersection_state['current_phase']
                    intersection_data['current_phase'] = phase_idx + 1  # Convert to 1-indexed
                    intersection_data['phase_name'] = self.ACTION_LOOKUP.get(phase_idx + 1, "Unknown")
                
                # Traffic pressure
                if 'pressure' in intersection_state:
                    intersection_data['pressure'] = intersection_state['pressure']
                
                # Vehicle counts
                if 'lane_vehicle_count' in intersection_state:
                    counts = intersection_state['lane_vehicle_count']
                    directions = ["North", "East", "South", "West"]
                    lanes_per_direction = len(counts) // 4
                    
                    vehicle_counts = {}
                    for i, direction in enumerate(directions):
                        start_idx = i * lanes_per_direction
                        end_idx = start_idx + lanes_per_direction
                        direction_counts = counts[start_idx:end_idx]
                        vehicle_counts[direction] = sum(direction_counts)
                    
                    intersection_data['vehicle_counts'] = vehicle_counts
                
                # Waiting vehicles
                if 'lane_waiting_vehicle_count' in intersection_state:
                    waiting_counts = intersection_state['lane_waiting_vehicle_count']
                    
                    waiting_vehicles = {}
                    for i, direction in enumerate(directions):
                        start_idx = i * lanes_per_direction
                        end_idx = start_idx + lanes_per_direction
                        direction_waiting = waiting_counts[start_idx:end_idx]
                        waiting_vehicles[direction] = sum(direction_waiting)
                    
                    intersection_data['waiting_vehicles'] = waiting_vehicles
                
                # Lane details
                lane_info = []
                if 'lane_vehicle_count' in intersection_state and 'lane_waiting_vehicle_count' in intersection_state:
                    counts = intersection_state['lane_vehicle_count']
                    waiting = intersection_state['lane_waiting_vehicle_count']
                    
                    for i, (count, wait) in enumerate(zip(counts, waiting)):
                        direction = directions[i // lanes_per_direction]
                        lane = i % lanes_per_direction + 1
                        lane_info.append({
                            "direction": direction,
                            "lane": lane,
                            "count": count,
                            "waiting": wait
                        })
                
                intersection_data['lane_info'] = lane_info
                intersections[str(intersection_id)] = intersection_data
        
        state_dict['intersections'] = intersections
        
        return state_dict

    def get_last_info(self):
        """
        Get the info dictionary from the last step.
        
        Returns:
            dict: Info dictionary from the last step, or None if no step has been taken
        """
        return self.last_info

# Traffic Control Guide for reference
GUIDE = """
### Traffic Signal Control Instructions

In this task, you are controlling traffic signals at intersections to optimize traffic flow. 
Your goal is to reduce congestion, minimize waiting times, and improve overall traffic efficiency.

---

#### Signal Phases and Their Meaning
- **WSES (Phase 1)**: West-East Straight traffic gets green light
- **NSSS (Phase 2)**: North-South Straight traffic gets green light
- **WLEL (Phase 3)**: West-East Left-turn traffic gets green light
- **NLSL (Phase 4)**: North-South Left-turn traffic gets green light

---

#### Your Goal
Select the optimal signal phase at each step to minimize traffic congestion, reduce waiting times,
and maximize throughput of vehicles through the intersection.

---

#### Traffic Metrics
- **Traffic Pressure**: Imbalance between incoming and outgoing traffic
- **Queue Length**: Number of vehicles waiting at the intersection
- **Waiting Vehicles**: Vehicles that are stopped (speed near zero)
- **Throughput**: Number of vehicles successfully passing through the intersection

---

#### Controls
Use these outputs to select a signal phase:
- `1`: Set signal to **WSES** (West-East Straight)
- `2`: Set signal to **NSSS** (North-South Straight)
- `3`: Set signal to **WLEL** (West-East Left-turn)
- `4`: Set signal to **NLSL** (North-South Left-turn)

---

#### Rewards
- **Traffic Pressure**: Each unit of traffic pressure costs -0.25
- **Queue Length**: Each waiting vehicle costs -0.25
- **Throughput**: Each vehicle passing through gives +0.5
- **Invalid Action**: Selecting an invalid action costs -1.0

---

Enjoy optimizing traffic flow!
"""

if __name__ == '__main__':
    # Test the environment
    env = TrafficControlEnv(num_intersections=2)  # Test with 2 intersections
    observation = env.reset(seed=42)
    print(observation)
    
    # Take a few random steps
    for i in range(10):
        if env.num_intersections == 1:
            action = np.random.randint(1, 5)  # Random action between 1 and 4
        else:
            # Generate a random action for each intersection
            action = [np.random.randint(1, 5) for _ in range(env.num_intersections)]
            
        observation, reward, done, info = env.step(action)
        print(f"\nStep {i+1}, Action: {action}, Reward: {reward:.2f}")
        print(info["text_observation"])
        if done:
            break
    
    print("\nEnvironment test complete.")