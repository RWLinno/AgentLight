from typing import Optional, List, Tuple, Any, Dict, Union
import numpy as np
import re
import os
import json
import time
from .base import BaseDiscreteActionEnv
from .utils.cityflow_env import CityFlowEnv
from .utils.my_utils import get_state_detail, state2text

class CityFlowAdapter(BaseDiscreteActionEnv):
    """
    Adapter class to make CityFlowEnv compatible with the BaseDiscreteActionEnv interface.
    
    This adapter wraps the CityFlowEnv environment and provides an interface for
    LLM-based controllers to interact with traffic intersections. It can represent
    a single intersection in a multi-intersection environment.
    
    Each intersection has its own state, actions, and rewards, allowing for
    decentralized control where different LLMs can control different intersections.
    """
    
    # Direction and location mappings for better state descriptions
    DIRECTIONS = {"N": "North", "S": "South", "E": "East", "W": "West"}
    MOVEMENTS = {"T": "through", "L": "left-turn", "R": "right-turn"}
    PHASES_4 = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
    PHASES_8 = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
    
    # Detailed phase explanations
    PHASE_EXPLANATIONS = {
        "NTST": "Northern and southern through lanes",
        "NLSL": "Northern and southern left-turn lanes",
        "NTNL": "Northern through and left-turn lanes",
        "STSL": "Southern through and left-turn lanes",
        "ETWT": "Eastern and western through lanes",
        "ELWL": "Eastern and western left-turn lanes",
        "ETEL": "Eastern through and left-turn lanes",
        "WTWL": "Western through and left-turn lanes"
    }
    
    def __init__(self, path_to_log: str, path_to_work_directory: str, 
                 dic_traffic_env_conf: Dict, dic_path: Dict, intersection_id: int = 0):
        """
        Initialize the CityFlowAdapter.
        
        Args:
            path_to_log: Path to log directory
            path_to_work_directory: Path to work directory containing config files
            dic_traffic_env_conf: Traffic environment configuration dictionary
            dic_path: Dictionary of paths for various files
            intersection_id: ID of the intersection this adapter controls (default: 0)
        """
        super().__init__()
        
        # Store which intersection this adapter controls
        self.intersection_id = intersection_id
        
        # Ensure required configuration keys are present
        self._add_missing_config_keys(dic_traffic_env_conf)
        
        # Initialize the CityFlow environment
        self.env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=path_to_work_directory,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path
        )
        
        # Set up phase information
        self._setup_phase_information(dic_traffic_env_conf)
        
        # Initialize state tracking
        self._current_states = None
        self._done = False
        self.cumulative_reward = 0
        self.rewards = []
        self.last_action = None
        
        # Store road information for detailed state rendering
        self.intersection_dict = self.env.intersection_dict
        
        # Create individual adapters for each intersection if this is the main adapter
        self.intersection_adapters = []
        if intersection_id == 0 and dic_traffic_env_conf["NUM_INTERSECTIONS"] > 1:
            # Create adapters for all other intersections
            for i in range(1, dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                self.intersection_adapters.append(
                    CityFlowAdapter(path_to_log, path_to_work_directory, 
                                   dic_traffic_env_conf, dic_path, i)
                )
    
    def _add_missing_config_keys(self, dic_traffic_env_conf: Dict):
        """Add default values for missing configuration keys"""
        defaults = {
            "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
            "TOP_K_ADJACENCY": 4,
            "OBS_LENGTH": 100,
            "MIN_ACTION_TIME": 10,
            "YELLOW_TIME": 5,
            "INTERVAL": 1.0,
            "NUM_ROW": 3,
            "NUM_COL": 4,
            "MODEL_NAME": "LLM",
            "LIST_MODEL_NEED_TO_UPDATE": []
        }
        
        for key, default_value in defaults.items():
            if key not in dic_traffic_env_conf:
                dic_traffic_env_conf[key] = default_value
    
    def _setup_phase_information(self, dic_traffic_env_conf: Dict):
        """Set up action space parameters based on phases configuration"""
        # Set up the phase list
        if isinstance(dic_traffic_env_conf.get("PHASE"), dict):
            # Handle dictionary-based phase configuration
            num_phases = len(dic_traffic_env_conf["PHASE"])
            self.phase_list = dic_traffic_env_conf.get("PHASE_LIST", None)
        else:
            # Handle list-based phase configuration
            num_phases = len(dic_traffic_env_conf.get("PHASE", []))
            self.phase_list = None
            
            # Handle 4-phase vs 8-phase systems
            if num_phases == 4:
                self.phase_list = list(self.PHASES_4.keys())
                self.phase_mapping = self.PHASES_4
            elif num_phases == 8:
                self.phase_list = list(self.PHASES_8.keys())
                self.phase_mapping = self.PHASES_8
            else:
                # Default to 4-phase system
                self.phase_list = list(self.PHASES_4.keys())
                self.phase_mapping = self.PHASES_4
        
        # Set up action space
        self.ACTION_SPACE = type('', (), {
            'n': num_phases,
            'start': 1  # Actions start from 1 since 0 is invalid action
        })()
        
        # Set up action lookup
        self.ACTION_LOOKUP = {}
        for i in range(num_phases):
            if self.phase_list and i < len(self.phase_list):
                phase_name = self.phase_list[i]
                phase_desc = self.PHASE_EXPLANATIONS.get(phase_name, "")
                self.ACTION_LOOKUP[i+1] = f"Phase {i+1}: {phase_name} ({phase_desc})"
            else:
                self.ACTION_LOOKUP[i+1] = f"Phase {i+1}"
    
    def extract_action(self, text: str) -> int:
        """
        Extract action from text input.
        
        Args:
            text: Text containing the action
            
        Returns:
            The extracted action as an integer, or INVALID_ACTION if none found
        """
        try:
            # First, check for signal tags common in LLM outputs
            signal_matches = re.findall(r'<signal>(.*?)</signal>', text)
            if signal_matches:
                phase_code = signal_matches[-1]  # Use the last match if multiple
                if phase_code in self.phase_mapping:
                    return self.phase_mapping[phase_code] + 1  # +1 to match action space
            
            # Then, check for phase codes in text (e.g., "ETWT")
            for phase_name in self.phase_mapping:
                if phase_name in text:
                    return self.phase_mapping[phase_name] + 1  # +1 to match action space
            
            # Finally, try to extract a number from the text
            for word in text.split():
                if word.isdigit():
                    action = int(word)
                    if action in self.get_all_actions():
                        return action
            
            # If no valid action found, return invalid action
            return self.INVALID_ACTION
        except:
            return self.INVALID_ACTION
    
    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        
        Args:
            mode: Rendering mode
            seed: Random seed for environment initialization
            
        Returns:
            Initial observation
        """
        # Handle the random seed before resetting the environment
        if seed is not None:
            # Store the seed for potential use in the environment
            self._random_seed = seed
            
            # Set numpy's random seed
            np.random.seed(seed)
            
            # The CityFlowEnv uses np.random.randint internally for its seed
            # Since we've set np.random's seed, this will be influenced by our seed
        
        # Reset the base environment
        states = self.env.reset()
        self._current_states = states
        self._done = False
        self.cumulative_reward = 0
        self.rewards = []
        self.last_action = None
        
        # Reset all intersection adapters with different seeds derived from the main seed
        for i, adapter in enumerate(self.intersection_adapters):
            adapter._current_states = states
            adapter._done = False
            adapter.cumulative_reward = 0
            adapter.rewards = []
            adapter.last_action = None
            
            # If seed is provided, give each adapter a derived seed
            if seed is not None:
                # Create a unique but deterministic seed for each adapter
                adapter_seed = seed * 1000 + i + 1
                np.random.seed(adapter_seed)  # Set for each adapter
        
        return self.render()
    
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: The action to take for this intersection
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if action == self.INVALID_ACTION:
            return self.render(), self.PENALTY_FOR_INVALID, self._done, {"action_is_effective": False}
        
        # Create a list of actions for all intersections
        action_list = [0] * self.env.dic_traffic_env_conf["NUM_INTERSECTIONS"]
        
        # Set this intersection's action
        action_list[self.intersection_id] = action - 1  # Subtract 1 since CityFlow uses 0-based actions
        
        # If this is the main adapter (id 0), gather actions from other adapters
        if self.intersection_id == 0:
            for i, adapter in enumerate(self.intersection_adapters, 1):
                # Use the last known action for each adapter or default to 0
                if hasattr(adapter, 'last_action') and adapter.last_action is not None and adapter.last_action != self.INVALID_ACTION:
                    action_list[i] = adapter.last_action - 1
                    
        # Store this action for future reference
        self.last_action = action
            
        next_states, rewards, done, info = self.env.step(action_list)
        
        self._current_states = next_states
        self._done = done
        current_reward = rewards[self.intersection_id]
        self.rewards.append(current_reward)
        self.cumulative_reward += current_reward
        
        # Update all intersection adapters
        for i, adapter in enumerate(self.intersection_adapters, 1):
            adapter._current_states = next_states
            adapter._done = done
            adapter.rewards.append(rewards[i])
            adapter.cumulative_reward += rewards[i]
        
        return self.render(), rewards[self.intersection_id], done, {"action_is_effective": True}
    
    def success(self) -> bool:
        """
        Check if the current environment is successful.
        
        In traffic control, we interpret "success" as achieving positive reward.
        
        Returns:
            True if cumulative reward is positive, False otherwise
        """
        return self.cumulative_reward > 0
    
    def finished(self) -> bool:
        """
        Check if the current environment is finished.
        
        Returns:
            True if the environment is done, False otherwise
        """
        return self._done
    
    def render(self, mode: str = 'tiny_rgb_array') -> str:
        """
        Render the environment with detailed traffic state description for this intersection.
        
        Args:
            mode: Rendering mode (not used, exists for compatibility)
            
        Returns:
            Text description of the current traffic state
        """
        if self._current_states is None:
            return "Environment not initialized yet."
        
        # Get intersection details
        intersection_name = f"Intersection {self.intersection_id+1}"
        if self.intersection_id < len(self.env.list_intersection):
            inter_obj = self.env.list_intersection[self.intersection_id]
            intersection_name = f"Intersection {inter_obj.inter_name} (ID: {self.intersection_id+1})"
        
        state_txt = f"🚦 {intersection_name} Traffic State 🚦\n\n"
        
        try:
            # Get detailed state information
            if self.env.eng is not None and self.intersection_id < len(self.env.list_intersection):
                # Use the specialized state detail function from LLMLight utils
                inter_obj = self.env.list_intersection[self.intersection_id]
                inter_dict = self.env.intersection_dict.get(inter_obj.inter_name, {})
                
                # Get detailed state for each lane
                state_detail, state_incoming, avg_speed = get_state_detail(
                    roads=inter_dict.get("roads", {}), 
                    env=self.env
                )
                
                # Format state in a readable way
                state_txt += self._format_detailed_state(state_detail, self.phase_list)
            else:
                # Basic state representation if detailed state is not available
                state = self._current_states[self.intersection_id]
                state_txt += self._format_basic_state(state)
        
        except Exception as e:
            # Fallback to simple state representation if detailed state fails
            state_txt += f"Error getting detailed state: {str(e)}\n\n"
            state = self._current_states[self.intersection_id]
            state_txt += self._format_basic_state(state)
        
        # Add current phase information
        if hasattr(self, 'last_action') and self.last_action is not None and self.last_action != self.INVALID_ACTION:
            phase_idx = self.last_action - 1
            if 0 <= phase_idx < len(self.phase_list):
                phase_name = self.phase_list[phase_idx]
                phase_desc = self.PHASE_EXPLANATIONS.get(phase_name, "")
                state_txt += f"Current Signal Phase: {phase_name} ({phase_desc})\n\n"
        
        # Add performance metrics
        state_txt += "📊 Performance Metrics 📊\n"
        if len(self.rewards) > 0:
            state_txt += f"Current reward: {self.rewards[-1]:.2f}\n"
        state_txt += f"Cumulative reward: {self.cumulative_reward:.2f}\n"
        if hasattr(self.env, 'eng') and self.env.eng is not None:
            state_txt += f"Simulation time: {self.env.eng.get_current_time():.1f}s\n"
        
        return state_txt
    
    def _format_detailed_state(self, state_detail: Dict, phase_list: List[str]) -> str:
        """Format detailed state information into readable text"""
        state_txt = "🚗 Queue Information 🚗\n"
        
        # Organize data by phase
        for phase in phase_list:
            lane_1 = phase[:2]  # First two characters (e.g., "ET")
            lane_2 = phase[2:]  # Last two characters (e.g., "WT")
            
            if lane_1 in state_detail and lane_2 in state_detail:
                queue_len_1 = int(state_detail[lane_1]['queue_len'])
                queue_len_2 = int(state_detail[lane_2]['queue_len'])
                total_queue = queue_len_1 + queue_len_2
                
                phase_desc = self.PHASE_EXPLANATIONS.get(phase, "")
                state_txt += f"\nSignal: {phase} ({phase_desc})\n"
                state_txt += f"- Queue lengths: {queue_len_1} ({self.DIRECTIONS[lane_1[0]]}), "
                state_txt += f"{queue_len_2} ({self.DIRECTIONS[lane_2[0]]}), {total_queue} (Total)\n"
                
                # Add vehicle counts in segments
                for seg_idx in range(3):
                    seg_name = f"Segment {seg_idx+1}"
                    # Handle the case where segments 3 and 4 are combined
                    if seg_idx == 2:
                        vehicles_1 = state_detail[lane_1]['cells'][2] + state_detail[lane_1]['cells'][3] \
                            if len(state_detail[lane_1]['cells']) > 3 else state_detail[lane_1]['cells'][2]
                        vehicles_2 = state_detail[lane_2]['cells'][2] + state_detail[lane_2]['cells'][3] \
                            if len(state_detail[lane_2]['cells']) > 3 else state_detail[lane_2]['cells'][2]
                    else:
                        vehicles_1 = state_detail[lane_1]['cells'][seg_idx]
                        vehicles_2 = state_detail[lane_2]['cells'][seg_idx]
                    
                    total_vehicles = vehicles_1 + vehicles_2
                    state_txt += f"- {seg_name}: {vehicles_1} ({self.DIRECTIONS[lane_1[0]]}), "
                    state_txt += f"{vehicles_2} ({self.DIRECTIONS[lane_2[0]]}), {total_vehicles} (Total)\n"
        
        return state_txt
    
    def _format_basic_state(self, state: Dict) -> str:
        """Format basic state information into readable text"""
        state_txt = ""
        
        # Include current phase if available
        if 'cur_phase' in state:
            current_phase = state['cur_phase'][0]
            if self.phase_list and current_phase < len(self.phase_list):
                phase_name = self.phase_list[current_phase]
                phase_desc = self.PHASE_EXPLANATIONS.get(phase_name, "")
                state_txt += f"Current signal phase: {phase_name} ({phase_desc})\n\n"
            else:
                state_txt += f"Current signal phase: {current_phase}\n\n"
        
        # Include queue information if available
        if 'traffic_movement_pressure_queue' in state:
            queue_data = state['traffic_movement_pressure_queue']
            state_txt += "Queue lengths at different approaches:\n"
            directions = ["East", "South", "West", "North"]
            for i, direction in enumerate(directions):
                queue_length = queue_data[i] if i < len(queue_data) else 0
                state_txt += f"- {direction}: {queue_length} vehicles\n"
        
        return state_txt
    
    def get_all_actions(self) -> List[int]:
        """
        Get all valid actions.
        
        Returns:
            List of all valid actions
        """
        return list(range(1, self.ACTION_SPACE.n + 1))
    
    def get_state_for_prompt(self) -> Dict:
        """
        Get detailed state information suitable for creating prompts for LLMs.
        
        Returns:
            Dictionary with detailed state information
        """
        if not hasattr(self.env, 'eng') or self.env.eng is None:
            return {}
        
        try:
            # Get the intersection object and dictionary
            inter_obj = self.env.list_intersection[self.intersection_id]
            inter_dict = self.env.intersection_dict.get(inter_obj.inter_name, {})
            
            # Get detailed state information
            state_detail, state_incoming, avg_speed = get_state_detail(
                roads=inter_dict.get("roads", {}), 
                env=self.env
            )
            
            return {
                "state": state_detail,
                "state_incoming": state_incoming,
                "avg_speed": avg_speed,
                "phase_list": self.phase_list,
                "current_time": self.env.eng.get_current_time()
            }
        except Exception as e:
            print(f"Error getting detailed state: {str(e)}")
            return {}
    
    def copy(self) -> 'CityFlowAdapter':
        """
        Create a deep copy of the environment.
        
        Note: CityFlow doesn't support deep copying, so we create a new instance
        with the same configuration and state.
        
        Returns:
            A new CityFlowAdapter instance with the same configuration and state
        """
        new_env = CityFlowAdapter(
            self.env.path_to_log,
            self.env.path_to_work_directory,
            self.env.dic_traffic_env_conf,
            self.env.dic_path,
            self.intersection_id
        )
        new_env._current_states = self._current_states
        new_env._done = self._done
        new_env.cumulative_reward = self.cumulative_reward
        new_env.rewards = self.rewards.copy() if self.rewards else []
        new_env.last_action = self.last_action
        
        # Copy intersection adapters if this is the main adapter
        if self.intersection_id == 0:
            new_env.intersection_adapters = [adapter.copy() for adapter in self.intersection_adapters]
            
        return new_env