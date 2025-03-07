import torch
import re
from collections import defaultdict
import os
import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from ragen.utils import set_seed
from ragen.utils.plot import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import numpy as np

# Import get_state_detail from utils if available
try:
    from LLMLight.utils.my_utils import get_state_detail
except ImportError:
    try:
        from utils.my_utils import get_state_detail
    except ImportError:
        try:
            # Another possible location
            from ragen.utils.my_utils import get_state_detail
        except ImportError:
            get_state_detail = None
            print("Warning: get_state_detail function could not be imported")

# Add phase dictionaries similar to chatgpt.py
four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
eight_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3, 'WTWL': 4, 'ETEL': 5, 'STSL': 6, 'NTNL': 7}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}

# Phase explanation dictionary similar to chatgpt.py
phase_explanation_dict_detail = {
    "NTST": "- NTST: Northern and southern through lanes.",
    "NLSL": "- NLSL: Northern and southern left-turn lanes.",
    "NTNL": "- NTNL: Northern through and left-turn lanes.",
    "STSL": "- STSL: Southern through and left-turn lanes.",
    "ETWT": "- ETWT: Eastern and western through lanes.",
    "ELWL": "- ELWL: Eastern and western left-turn lanes.",
    "ETEL": "- ETEL: Eastern through and left-turn lanes.",
    "WTWL": "- WTWL: Western through and left-turn lanes."
}


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    logging: dict
    num_gpus: int
    no_think_rl: bool=False
    state_masking: bool=False
    start_state_marker: str="<start-state>"
    end_state_marker: str="<end-state>"

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        env_class,
        env,
        config: GenerationConfig,
        logger: Tracking,
        is_validation: bool = False,
        intersection=None,
    ):
        """
        Initialize the LLMGenerationManager.
        
        Args:
            tokenizer: The tokenizer for processing text
            actor_rollout_wg: The actor model for rollouts
            env_class: The environment class to use
            env: The environment instance
            config: Configuration parameters
            logger: Tracking object for logging metrics
            is_validation: Whether this is for validation
            intersection: Optional intersection data for traffic control tasks
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.env_class = env_class
        self.env = env
        self.config = config
        self.logger = logger
        self.is_validation = is_validation
        
        # Store intersection data if provided (for traffic control tasks)
        self.intersection = intersection
        self.roads = None
        self.inter_name = None
        
        # Initialize roads data if intersection information is provided
        if intersection is not None:
            print("Initializing with provided intersection data")
            if isinstance(intersection, dict) and "roads" in intersection:
                self.roads = copy.deepcopy(intersection["roads"])
                print(f"Successfully stored roads data with {len(self.roads)} roads")
            
            # Store intersection name if available
            if "name" in intersection:
                self.inter_name = intersection["name"]
                print(f"Using intersection: {self.inter_name}")
                
        # This will be used for state processing
        self.phases = four_phase_list
        # Default to 4-phase traffic control unless specified
        self.phase_num = 4
        
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        if intersection and "roads" in intersection:
            self.roads = copy.deepcopy(intersection["roads"])
            for r in self.roads:
                if "location" in self.roads[r] and "length" in self.roads[r]:
                    self.length_dict[self.roads[r]["location"]] = int(self.roads[r]["length"])
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        # Ensure responses is always a list of strings
        if isinstance(responses, str):
            responses = [responses]
        elif not isinstance(responses, list):
            raise TypeError(f"Expected responses to be a string or list of strings, got {type(responses)}")
            
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    @staticmethod
    def _process_answer_tag(responses_str):
        """
        Process a list of response strings to extract the answer from within <answer> tags,
        similar to how the chatgpt.py implementation extracts signal text.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing answer tags
            
        Returns:
            List[str]: Processed responses containing only the content within answer tags
        """
        def process_single_response(resp):
            # Define the pattern to match the answer tag
            answer_pattern = r'<answer>(.*?)</answer>'
            
            # Find all matches in the response text
            matches = re.findall(answer_pattern, resp, re.DOTALL)
            
            # If no matches found, return the original response
            if not matches:
                return resp
            
            # Extract the answer
            answer = matches[-1].strip()  # Use the last match, similar to chatgpt.py
            
            return f"<answer>{answer}</answer>"
        
        # Process each response string
        return [process_single_response(resp) for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor, envs: List[Any]) -> Tuple[torch.Tensor, List[str]]:
        """
        Process responses to extract actions and handle various cases like reward hacking.
        Similar to the extraction logic in chatgpt.py's choose_action method.
        
        Args:
            responses: Tensor containing model responses
            envs: List of environment instances
            
        Returns:
            Processed response tensor and response strings
        """
        # Decode the tensor responses to strings
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Process answer tags to extract the actual answers
        responses_str = self._process_answer_tag(responses_str)
        
        # Handle state masking if enabled (prevents reward hacking)
        if self.config.state_masking:
            # Escape special characters in markers for regex
            start_marker = re.escape(self.config.start_state_marker)
            end_marker = re.escape(self.config.end_state_marker)
            hack_pattern = f'{start_marker}[\\s\\S]*?{end_marker}'
            
            # Check if any responses contain hacking attempts
            hacked = [resp for resp in responses_str if re.search(hack_pattern, resp, re.DOTALL)]
            if hacked:
                print(f"[WARNING] HACKED RESPONSES: {len(hacked)} detected")
            
            # Remove any state marker sections that shouldn't be there
            responses_str = [re.sub(hack_pattern, '', resp, re.DOTALL) for resp in responses_str]

        # Handle no_think_rl mode (only keeps action)
        if self.config.no_think_rl:
            # Extract actions from the responses
            try:
                actions, _ = self.env_class.postprocess_predictions(envs, responses_str)
                # Format responses to only include the action
                responses_str = [f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" 
                             for idx, action in enumerate(actions)]
                print("RESPONSES:", responses_str)
            except Exception as e:
                print(f"Error in postprocessing actions: {e}")
                # If there's an error, keep the original responses
                pass
        
        # Re-tokenize the processed strings
        responses = self._batch_tokenize(responses_str)
        
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        if self.config.state_masking:
            start_marker = self.config.start_state_marker
            end_marker = self.config.end_state_marker
            
            # Create inner versions by adding 'inner_' prefix
            inner_start = f"<inner_{start_marker[1:]}"
            inner_end = f"<inner_{end_marker[1:]}"
            
            # Replace any existing markers with inner versions
            next_obs = [re.sub(re.escape(start_marker), inner_start, obs) for obs in next_obs]
            next_obs = [re.sub(re.escape(end_marker), inner_end, obs) for obs in next_obs]
            
            # Wrap with state markers
            next_obs = [f"{start_marker}{obs}{end_marker}" for obs in next_obs]
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            
        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            next_obs_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    

    
    # NOTE: xc: 在进行LLM_Loop的时候，需要用step更新环境：如，在generation文件中原作者也有写到，如果觉得写得过于复杂，可以重新简单定义
        """
        actions = res_to_action(xxxx) #这里的action应该是一个list，包含了所有路口response得到的action，#envs是多CPU多开的多个环境，actions是同步收集的多个list,用来加速训练
        for env, action in zip(envs, actions):
            env.step(action)
        """
    def generate_traffic_prompt(self, env, target_intersection=None):
        """
        Generate a prompt for the traffic control task based on current state.
        
        Args:
            env: Environment or intersection object
            target_intersection: Optional specific intersection to target
            
        Returns:
            Tuple of (prompt_text, is_default_prompt)
        """
        attempted_methods = ["direct state detail retrieval"]
        print("\n===== GENERATING TRAFFIC PROMPT =====")
        
        try:
            #            # Try using the imported get_state_detail function
                    print("Using imported get_state_detail function")
                    state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)
                    print(f"Successfully retrieved state using imported get_state_detail")
                    
                    # Check if state is empty (similar to chatgpt.py flow_num check)
                    flow_num = 0
                    for road in state:
                        flow_num += state[road]["queue_len"] + sum(state[road]["cells"])
                    
                    if flow_num == 0:
                        print("State is empty (no vehicles detected)")
                        return "No vehicles detected. Use the default signal: ETWT, please reply <answer>ETWT</answer>, do not reply with any other text", True
                    
                    # Convert state to a readable table format
                    state_txt = self._state_to_table(state, env)
                    
                    # Generate a well-formatted prompt with the state information
                    prompt = state_txt
                    return prompt, False
            
        except Exception as e:
            print(f"Error with direct state detail retrieval: {e}")
            print("=================================================\n")
            
        
        # If we made it here, we couldn't get the state
        print(f"\n===== COULD NOT RETRIEVE TRAFFIC STATE =====")
        print(f"Attempted methods: {', '.join(attempted_methods)}")
        print("Using default signal: ETWT")
        print("=================================================\n")
        
        return "Could not retrieve traffic state. Use the default signal: ETWT", True

    def _direct_engine_check(self, env):
        """Directly check the engine for traffic."""
        print("\n===== TRYING DIRECT ENGINE ACCESS =====")
        
        # Try to get lane vehicle counts directly from engine
        if hasattr(env.eng, 'get_lane_waiting_vehicle_count') and hasattr(env.eng, 'get_lane_vehicles'):
            try:
                lane_queues = env.eng.get_lane_waiting_vehicle_count()
                lane_vehicles = env.eng.get_lane_vehicles()
                
                print(f"Lane queues: {lane_queues}")
                print(f"Lane vehicles: {lane_vehicles}")
                
                # If there are vehicles or queues, there's traffic
                total_vehicles = sum([len(vehicles) for vehicles in lane_vehicles.values()])
                total_queue = sum(lane_queues.values())
                
                if total_vehicles > 0 or total_queue > 0:
                    print(f"Traffic detected: {total_vehicles} vehicles, {total_queue} in queue")
                    return True
                else:
                    print("No traffic detected in the environment")
                    return False
            except Exception as e:
                print(f"Error accessing engine: {str(e)}")
                return False
        
        print("Engine methods not available")
        return False

    def _check_traffic_flow(self, state):
        """Check if there's any traffic flow in the state."""
        flow_num = 0
        for road in state:
            if isinstance(state[road], dict) and "queue_len" in state[road] and "cells" in state[road]:
                flow_num += state[road]["queue_len"] + sum(state[road]["cells"])
                print(f"Lane '{road}': queue_len={state[road]['queue_len']}, cells={state[road]['cells']}, flow={state[road]['queue_len'] + sum(state[road]['cells'])}")
        
        print(f"Total traffic flow detected: {flow_num}")
        return flow_num

    def _state_to_table(self, state, env):
        """
        Format state dictionary into a structured table, similar to state2table in chatgpt.py.
        
        Args:
            state: Dictionary containing state information
            env: The environment or intersection
            
        Returns:
            Formatted state text
        """
        # Early return with error message if state is None or not a dict
        if state is None:
            return "No state information available. Default signal: ETWT"
        
        if not isinstance(state, dict):
            return f"Invalid state format. Default signal: ETWT"
            
        # Special case: check if this is a traffic pressure format
        if 'traffic_movement_pressure_queue' in state:
            # Convert traffic pressure format to standard state format
            print("\n===== TRANSFORMING TRAFFIC_MOVEMENT_PRESSURE_QUEUE IN _state_to_table =====")
            pressure_data = state['traffic_movement_pressure_queue']
            if isinstance(pressure_data, dict):
                print(f"Processing pressure_data with {len(pressure_data)} keys")
                print(f"Keys: {', '.join(list(pressure_data.keys())[:15])}")
                
                transformed_state = {}
                for lane_key, lane_data in pressure_data.items():
                    if isinstance(lane_key, str) and len(lane_key) == 2:
                        print(f"Transforming lane '{lane_key}'")
                        if isinstance(lane_data, dict):
                            print(f"  Lane data keys: {', '.join(list(lane_data.keys()))}")
                        else:
                            print(f"  Lane data is not a dictionary: {type(lane_data).__name__}")
                            
                        # Create normalized lane data
                        transformed_state[lane_key] = {
                            'queue_len': lane_data.get('queue_len', 0) if isinstance(lane_data, dict) else 0,
                            'cells': lane_data.get('cells', [0, 0, 0]) if isinstance(lane_data, dict) else [0, 0, 0]
                        }
                        print(f"  Created entry: {lane_key}: {transformed_state[lane_key]}")
                
                # If we transformed successfully, use the new state
                if transformed_state:
                    print(f"Successfully transformed state to have {len(transformed_state)} lanes")
                    state = transformed_state
                else:
                    print("Failed to transform any lanes from traffic_movement_pressure_queue")
                    # Return a default message if transformation failed
                    return "No valid traffic data available. Default signal: ETWT"
            else:
                print(f"pressure_data is not a dictionary, it's a {type(pressure_data).__name__}")
                return "Invalid traffic data format. Default signal: ETWT"
        
        # Generate table text in the same format as chatgpt.py's state2table method
        state_txt = ""
        
        # Check if we have enough lane information for traffic control
        lane_keys = [k for k in state.keys() if isinstance(k, str) and len(k) == 2 
                    and k[0] in 'NSEW' and k[1] in 'TLR']
        
        if not lane_keys:
            print("No valid lane information found in state.")
            return "No valid traffic data found. Default signal: ETWT"
            
        # Check which phases we can generate
        valid_phases = []
        for phase_name, phase_code in self.phases.items():
            lane1 = phase_name[:2]
            lane2 = phase_name[2:]
            if lane1 in lane_keys and lane2 in lane_keys:
                valid_phases.append(phase_name)
                
        if not valid_phases:
            print("No valid phase information can be generated.")
            return "Cannot generate valid phase information. Default signal: ETWT"
        
        # Format traffic state information
        for p in valid_phases:
            lane_1 = p[:2]  # First two chars e.g. "NT"
            lane_2 = p[2:]  # Last two chars e.g. "ST"
            
            # Get queue lengths
            queue_len_1 = 0
            queue_len_2 = 0
            if lane_1 in state and "queue_len" in state[lane_1]:
                queue_len_1 = int(state[lane_1]['queue_len'])
            if lane_2 in state and "queue_len" in state[lane_2]:
                queue_len_2 = int(state[lane_2]['queue_len'])
            
            # Get cell information
            cells_1 = [0, 0, 0, 0]
            cells_2 = [0, 0, 0, 0]
            
            if lane_1 in state and "cells" in state[lane_1]:
                cells = state[lane_1]['cells']
                for i in range(min(len(cells), 4)):
                    cells_1[i] = cells[i]
                    
            if lane_2 in state and "cells" in state[lane_2]:
                cells = state[lane_2]['cells']
                for i in range(min(len(cells), 4)):
                    cells_2[i] = cells[i]
            
            # Format segment information
            seg_1_lane_1 = cells_1[0]
            seg_2_lane_1 = cells_1[1]
            seg_3_lane_1 = cells_1[2] + (cells_1[3] if len(cells_1) > 3 else 0)
            
            seg_1_lane_2 = cells_2[0]
            seg_2_lane_2 = cells_2[1]
            seg_3_lane_2 = cells_2[2] + (cells_2[3] if len(cells_2) > 3 else 0)
            
            # Get location names
            location_1 = f"{location_dict[lane_1[0]]}" if lane_1[0] in location_dict else lane_1[0]
            location_2 = f"{location_dict[lane_2[0]]}" if lane_2[0] in location_dict else lane_2[0]
            
            # Format state text for this phase
            state_txt += (f"Signal: {p}\n"
                       f"Allowed lanes: {lane_1} and {lane_2} ({location_1} {direction_dict[lane_1[1]]} and {location_2} {direction_dict[lane_2[1]]})\n"
                       f"- Early queued: {queue_len_1} ({location_1}), {queue_len_2} ({location_2}), {queue_len_1 + queue_len_2} (Total)\n"
                       f"- Segment 1: {seg_1_lane_1} ({location_1}), {seg_1_lane_2} ({location_2}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                       f"- Segment 2: {seg_2_lane_1} ({location_1}), {seg_2_lane_2} ({location_2}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                       f"- Segment 3: {seg_3_lane_1} ({location_1}), {seg_3_lane_2} ({location_2}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")
        
        # Create a prompt exactly like getPrompt in chatgpt.py
        if state_txt:
            prompt = (
                "# Traffic Signal Control Task\n\n"
                "A crossroad connects two roads: the north-south and east-west. The traffic light is located at "
                "the intersection of the two roads. The north-south road is divided into two sections by the intersection: "
                "the north and south. Similarly, the east-west road is divided into the east and west. Each section "
                "has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. "
                "Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. "
                "In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. "
                "Early queued vehicles have arrived at the intersection and await passage permission. Approaching "
                "vehicles will arrive at the intersection in the future.\n\n"                
                "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                "specific lanes. The state of the intersection is listed below. It describes:\n"
                "- The group of lanes relieving vehicles' flow under each traffic light phase.\n"
                "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                "<STATE>"
            ) + state_txt + (
                "</STATE>"
                "\nPlease answer:\n"
                "Which is the most effective traffic signal that will most significantly improve the traffic "
                "condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?\n\n"
                "<NOTE>\n"
                "The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant "
                "impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to "
                "consider vehicles in distant segments since they are unlikely to reach the intersection soon.</NOTE>\n\n"                
                "<REQUIREMENTS>\n"
                "- You can only choose one of the signals listed above.\n"
                "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                "- Your choice can only be given after finishing the analysis.\n"
                "- Your choice must be identified by the tag: <answer>YOUR_CHOICE</answer>!!! </REQUIREMENTS>"
            )
            return prompt
        else:
            return "No valid traffic information available. Default signal: ETWT"

    def generate_messages(self, envs: List[Any]) -> List[str]:
        """
        Generate LLM prompts for traffic signal control using environment states.
        
        Args:
            envs: List of traffic environment instances
            
        Returns:
            List of formatted prompt strings
        """
        messages_list = []
        
        for env in envs:
            # Generate prompt for this specific environment
            prompt = self.generate_traffic_prompt(env)
            messages_list.append(prompt)
        
        return messages_list
    
    def run_llm_loop(self, envs: List[Any],
                    batch: DataProto,
                    initial_input_ids: torch.Tensor,
                    output_dir: str,
                    global_steps: int) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop for traffic control, processing multiple intersections per environment.
        
        Args:
            envs: List of environment instances
            initial_input_ids: Initial input tensor
            output_dir: Directory to save outputs
            global_steps: Current global step count
            
        Returns:
            Tuple of final output dictionaries
        """
        
        print(f"\n===== CHECKING ENVIRONMENTS FORMAT =====")
        
        # Handle the case where a single environment is passed directly (not in a list)
        # This would be the case when passed from OneLine.train
        print(f"Environments type: {type(envs).__name__}")
        print(f"Number of environments: {len(envs)}")
        
        # Check if the first item is a list and unwrap if necessary
        if len(envs) > 0 and isinstance(envs[0], list):
            print(f"Detected nested list structure - unwrapping")
            print(f"Ens[1] type: {type(envs[0]).__name__}, length: {len(envs[0])}")
            print(f"content of envs[0]: {envs[0]}")
            
        
        print("=======================================\n")
        # Initialize tracking variables
        batch_size = len(envs)
        tensor_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses', 'prompts']
                        
        trajectory = self._setup_visualization()
        
        # Create an initial input tensor if none provided
        if initial_input_ids is None:
            initial_input_ids = torch.ones((batch_size, 1), dtype=torch.long)
        
        # Setup tracking structures
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        # Track which environments are still active
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        
        # Create metadata for tracking
        meta_info = {}
       

        # Initialize 3D data structures for batch (envs * step * intersections)
        if 'input_ids' not in batch.batch:
            batch.batch['input_ids'] = [[] for _ in range(batch_size)]
            batch.batch['attention_mask'] = [[] for _ in range(batch_size)]
            batch.batch['position_ids'] = [[] for _ in range(batch_size)]
            batch.batch['prompts'] = [[] for _ in range(batch_size)]
            batch.batch['responses'] = [[] for _ in range(batch_size)]
            
        # Initialize or convert reward structure to avoid tensor append issues
        if 'reward' not in batch.non_tensor_batch:
            batch.non_tensor_batch['reward'] = [[] for _ in range(batch_size)]
        
        # Main loop for the number of turns
        for step in range(self.config.max_turns):
            # Break if no environments are active
            if not active_mask.sum():
                break
            
            
            # Get active environments
            envs = [env for mask, env in zip(active_mask, envs) if mask]
            active_indices = [i for i, mask in enumerate(active_mask) if mask]
            
            # Container for storing all actions for all environments
            all_env_actions = [[] for _ in range(len(envs))]
            all_env_responses = [[] for _ in range(len(envs))]
            all_env_responses_ids = [[] for _ in range(len(envs))]
            all_intersection_prompts = [[] for _ in range(len(envs))]
            all_input_ids = [[] for _ in range(len(envs))]
            all_attention_mask = [[] for _ in range(len(envs))]
            all_position_ids = [[] for _ in range(len(envs))]
            all_prompts = [[] for _ in range(len(envs))]
            all_responses = [[] for _ in range(len(envs))]
            all_reward = [[] for _ in range(len(envs))]
            total_reward = [[] for _ in range(len(envs))]


                            
            
            try:
                # Process each active environment
                for env_idx, env in zip(active_indices, envs):
                    env = envs[env_idx]
                    total_reward[env_idx] = 0.0
                    # Update roads data if not already stored
                    self._update_intersection_data(env)
                    print(f"---------------------------------env---------------------------------: {env}")
                    # Print more detailed info about the environment
                    print("\nEnvironment Details:")
                    print(f"Type: {type(env).__name__}")

                    # Process each intersection in the environment
                    # Get the number of intersections in this environment
                    num_intersections = 12  # Default to one intersection if we can't determine
                    
                    # Try different methods to get intersection count
                    if hasattr(env, 'list_intersection'):
                        num_intersections = len(env.list_intersection)
                    elif hasattr(env, 'intersections'):
                        num_intersections = len(env.intersections)
                    elif hasattr(env, 'get_intersection_count'):
                        num_intersections = env.get_intersection_count()
                    

                   
                        
                        # Handle reward structure separately
                        if not isinstance(batch.non_tensor_batch['reward'][env_idx], list):
                            batch.non_tensor_batch['reward'][env_idx] = list(batch.non_tensor_batch['reward'][env_idx]) if hasattr(batch.non_tensor_batch['reward'][env_idx], '__iter__') else [batch.non_tensor_batch['reward'][env_idx]]
                        batch.non_tensor_batch['reward'][env_idx].append([0.0] * num_intersections)

                    # Container for intersection actions
                    intersection_actions = []
                    intersection_responses = []
                    intersection_responses_ids = []
                    intersection_prompts = []
                    intersection_penalties = []
                    # Process each intersection
                    for i in range(num_intersections):
                        intersection = None
                        if hasattr(env, 'list_intersection') and i < len(env.list_intersection):
                            intersection = env.list_intersection[i]
                            print(f"---------------------------------intersection---------------------------------: {intersection}")
                        elif hasattr(env, 'intersections') and i < len(env.intersections):
                            intersection = env.intersections[i]
                        
                        # Generate prompt for this intersection
                        target = intersection
                        
                        # Get the prompt
                        prompt = self.generate_traffic_prompt(target)
                        if prompt is None:
                            print(f"Warning: generate_traffic_prompt returned None for intersection {i}")
                            prompt = "No valid traffic information available. Default signal: ETWT", True
                            
                        print(f"Generated prompt for intersection {i}")
                        print(f"prompt: {prompt}")

                        intersection_prompts.append(prompt)
                        
                        # Handle prompt which may be a tuple (text, is_default)
                        if isinstance(prompt, tuple) and len(prompt) > 0:
                            prompt_text = prompt[0]  # Extract just the text part
                        else:
                            prompt_text = prompt  # Already a string
                        
                        # Convert prompt to tokens
                        try:
                            # Ensure we're using a string for tokenization
                            print(f"Tokenizing prompt of type: {type(prompt_text)}")
                            prompt_ids = self.tokenizer(
                                [prompt_text],  # Wrap in a list to ensure correct input format
                                padding='longest',
                                return_tensors='pt'
                            )
                        except Exception as e:
                            print(f"Error tokenizing prompt: {e}")
                            # Provide a simple fallback prompt in case of tokenization errors
                            prompt_text = "Traffic light control task. Default signal: ETWT"
                            prompt_ids = self.tokenizer(
                                [prompt_text],  # Also wrap the fallback in a list
                                padding='longest',
                                return_tensors='pt'
                            )
                        
                        # Create DataProto for generation
                        rolling_input = DataProto.from_dict({
                            'input_ids': prompt_ids['input_ids'],
                            'attention_mask': prompt_ids['attention_mask'],
                            'position_ids': torch.arange(prompt_ids['input_ids'].shape[1]).unsqueeze(0).expand_as(prompt_ids['input_ids'])
                        })
                        
                        
                        # Generate response for this intersection
                        retry_counter = 0
                        max_retries = 1
                        valid_response = False
                        signal_text = ""
                        
                        # Track the last response and its tokenized form
                        last_response_str = ""
                        last_response_ids = None
                        last_gen_output = None
                        
                        while retry_counter < max_retries and not valid_response:
                            try:
                                # Generate response
                                gen_output = self._generate_with_gpu_padding(rolling_input)
                                
                                # Update meta info
                                if not meta_info:
                                    meta_info.update(gen_output.meta_info)
                                
                                # Post-process the model response
                                response_ids, response_str = self._postprocess_responses(
                                    gen_output.batch['responses'], envs=[envs]
                                )
                                
                                # Store the last response
                                last_response_str = response_str[0] if response_str else ""
                                last_response_ids = response_ids[0] if response_ids.shape[0] > 0 else None
                                last_gen_output = gen_output
                                
                                print(f"response_str: {response_str}")
                                # Extract action using pattern matching, similar to chatgpt.py
                                signal_answer_pattern = r'<answer>(.*?)</answer>'
                                matches = re.findall(signal_answer_pattern, response_str[0], re.DOTALL)
                                print(f"matches: {matches}")
                                
                                if matches and matches[-1].strip():
                                    # Extract the action text (signal)
                                    signal_text = matches[-1].strip()
                                    
                                    # Convert to action code if needed
                                    action_code = None
                                    
                                    # Check if the signal text is directly in the action lookup
                                    if hasattr(target, 'ACTION_LOOKUP') and signal_text in target.ACTION_LOOKUP:
                                        action_code = target.ACTION_LOOKUP.index(signal_text)
                                    # Check for numeric signal (1-4)
                                    elif signal_text.isdigit():
                                        action_num = int(signal_text)
                                        # Adjust for 1-indexed input if needed
                                        if action_num >= 1 and action_num <= (len(target.ACTION_LOOKUP) if hasattr(target, 'ACTION_LOOKUP') else 4):
                                            action_code = action_num - 1
                                    else:
                                        # Try to find the action by name in available actions
                                        available_actions = target.ACTION_LOOKUP if hasattr(target, 'ACTION_LOOKUP') else list(four_phase_list.keys())
                                        
                                        # Check direct match
                                        for idx, act in enumerate(available_actions):
                                            if act in signal_text:
                                                action_code = idx
                                                break
                                    
                                    if action_code is not None:
                                        intersection_actions.append(action_code)
                                        intersection_responses.append(response_str[0])
                                        intersection_responses_ids.append(response_ids[0])
                                        valid_response = True
                                        print(f"Extracted action {action_code} from signal: {signal_text}")
                                        intersection_penalties.append(0)
                                    
                                    else:
                                        retry_counter += 1
                                        print(f"Invalid signal text: {signal_text}, retry {retry_counter}/{max_retries}")
                                else:
                                    retry_counter += 1
                                    print(f"No valid answer tag found, retry {retry_counter}/{max_retries}")
                            
                            except Exception as e:
                                retry_counter += 1
                                print(f"Error processing response (attempt {retry_counter}/{max_retries}): {e}")
                        
                        # If no valid response after retries, use the last invalid response
                        if not valid_response:
                            print("Using last invalid response after maximum retries")
                            intersection_penalties.append(-100)
                            # If we have a last response, use it
                            if last_response_ids is not None and last_gen_output is not None:
                                # Use the default action but keep the invalid response for context
                                default_action = 0  # ETWT/WSES is often the default
                                default_signal = target.ACTION_LOOKUP[default_action] if hasattr(target, 'ACTION_LOOKUP') else 'ETWT'
                                
                                intersection_actions.append(default_action)
                                intersection_responses.append(last_response_str)
                                intersection_responses_ids.append(last_response_ids)
                                
                                # Use the last gen_output
                                gen_output = last_gen_output
                                
                                print(f"Using default action: {default_action} with signal: {default_signal}")
                                #add reward penalty for invalid response



                                print(f"But keeping original response: {last_response_str[:50]}...")
                            else:
                                # If we don't have a last response, print the error
                                print(f"Error: No valid response after maximum retries")
                        
                        # Compose combined data for batch storage using helper functions (similar to _compose_final_output)
                        intersection_data = {}
                        
                        # Store the prompt and response separately
                        intersection_data['prompts'] = rolling_input.batch['input_ids']
                        intersection_data['responses'] = gen_output.batch['responses']
                        
                        # Combine input IDs
                        intersection_data['input_ids'] = torch.cat([
                            rolling_input.batch['input_ids'],
                            gen_output.batch['responses']
                        ], dim=1)
                        
                        # Create attention mask using tensor helper
                        intersection_data['attention_mask'] = torch.cat([
                            rolling_input.batch['attention_mask'],
                            self.tensor_fn.create_attention_mask(gen_output.batch['responses'])
                        ], dim=1)
                        
                        # Create position ids using tensor helper
                        intersection_data['position_ids'] = self.tensor_fn.create_position_ids(
                            intersection_data['attention_mask']
                        )
                        
        
                        # Store data in 3D format (envs * step * intersections)
                        # Ensure the intersection level is a list that we can set by index
                       
                        
                        # Now set the intersection data
                        all_input_ids[env_idx].append(intersection_data['input_ids'])
                        all_attention_mask[env_idx].append(intersection_data['attention_mask'])
                        all_position_ids[env_idx].append(intersection_data['position_ids'])
                        all_prompts[env_idx].append(intersection_data['prompts'])
                        all_responses[env_idx].append(intersection_data['responses'])
                    
                    # Assign the intersection actions to all_env_actions for this environment
                    all_env_actions[env_idx].extend(intersection_actions)  # Use extend for lists
                    all_env_responses[env_idx].extend(intersection_responses)
                    all_env_responses_ids[env_idx].extend(intersection_responses_ids)
                    all_intersection_prompts[env_idx].extend(intersection_prompts)
                
                # Execute actions in all environments
                next_observations = []  # Explicitly initialize as an empty list
                dones = []  # Explicitly initialize as an empty list
                print(f'intersection_penalties: {intersection_penalties}')
                for env_idx, env in zip(active_indices, envs):
                    env_actions = all_env_actions[env_idx]
                    
                    
                    print(f"ENV_STEP-----------------------")
                    # make sure the env_actions is a init list before step
                    print(f"env_actions: {env_actions}")
                    next_obs, reward, done, detail_reward = env.step(env_actions)
                    print(f"ENV_STEP_Finished--------------------------------")
                    # Check if reward is a scalar or array
                    for r,p in zip(detail_reward, intersection_penalties):
                        all_reward[env_idx].append(r+p)
                    print(f"batch.non_tensor_batch['reward']: {batch.non_tensor_batch['reward']}")
                    total_reward[env_idx] += reward
                

            except Exception as e:
                print(f"Error in LLM loop step {step}: {e}")
                # Continue to next step if there's an error
                continue
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        print(f"total_reward: {total_reward}")

        #final reward= all_reward + total_reward
        all_reward[env_idx] = [x + total_reward for x in all_reward[env_idx]]
        print(f"all_reward: {all_reward}")

        # Update the existing batch directly instead of creating a new DataProto object
        print(f"all_input_ids: {all_input_ids}")
        print(f"type of all_input_ids: {type(all_input_ids)}")
        print(f"instance of all_input_ids: {isinstance(all_input_ids, torch.Tensor)}")

        # NOTE ， 这里的转换可能是不必要的，错误的
        all_input_ids = torch.tensor(all_input_ids)
        all_attention_mask = torch.tensor(all_attention_mask)
        all_position_ids = torch.tensor(all_position_ids)
        all_prompts = torch.tensor(all_prompts)
        all_responses = torch.tensor(all_responses)

        final_output = DataProto.from_dict(tensors={
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'position_ids': all_position_ids,
            'prompts': all_prompts,
            'responses': all_responses
        }, non_tensors={
            'reward': all_reward
        })
        # for env_idx in enumerate(active_indices):
        #         for key, data_list in [
        #             ('input_ids', all_input_ids),
        #             ('attention_mask', all_attention_mask),
        #             ('position_ids', all_position_ids),
        #             ('prompts', all_prompts),
        #             ('responses', all_responses)
        #         ]:
        #             batch.batch[key][env_idx].extend(data_list[env_idx])
                
        #         # Update non-tensor batch data (reward)
    
        #         batch.non_tensor_batch['reward'][env_idx].extend(all_reward[env_idx])

        # Save trajectory for visualization
        # if trajectory:
        #     self._save_trajectory(trajectory, output_dir, global_steps)
        
        # Return the final output
        #return self._compose_final_output(original_left_side, original_right_side, meta_info), total_reward
        print(f"final_output: {final_output}")
        return final_output, total_reward
    def _setup_visualization(self) -> List[Dict]:
        """Setup visualization tracking if enabled."""
        if not self.config.logging.log_images:
            return None
        return [defaultdict(list) for _ in range(self.config.logging.log_n_image_per_batch)]

    def _update_trajectory(self, trajectory: List[Dict], 
                         envs: List[Any], responses: List[str], active_mask: torch.Tensor):
        """Update visualization trajectory if enabled."""
        if not trajectory:
            return
        n_visualize = self.config.logging.log_n_image_per_batch
        for idx, (env, active) in enumerate(zip(envs[:n_visualize], active_mask[:n_visualize])):
            if active:
                #trajectory[idx]['state'].append(env.render('rgb_array'))
                print(f"we don need to update trajectory")
            
        for idx, (response, env, active) in enumerate(zip(responses[:n_visualize], 
                                                envs[:n_visualize],
                                                active_mask[:n_visualize])):
            if active:
                parsed = parse_llm_output(response, strategy="raw")
                
                trajectory[idx]['answer'].append(response)
                trajectory[idx]['parsed_response'].append(parsed)

    def _save_trajectory(self, trajectory: List[Dict], 
                        output_dir: str, global_steps: int):
        """Save trajectory visualization if enabled."""
        if not trajectory:
            return
            
        save_step_size = self.config.logging.log_image_step_size
        if not global_steps % save_step_size or self.is_validation:
            os.makedirs(output_dir, exist_ok=True)
            filenames = save_trajectory_to_output(trajectory, save_dir=output_dir)
            if 'wandb' in self.logger.logger:
                for filename in filenames:
                    self.logger.logger['wandb'].save(filename)


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def initialize_roads(self, intersection_data):
        """
        Initialize or update roads data from intersection information.
        
        Args:
            intersection_data: Dictionary containing intersection information with roads data
        """
        if intersection_data and "roads" in intersection_data:
            self.roads = copy.deepcopy(intersection_data["roads"])
            self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
            for r in self.roads:
                if "location" in self.roads[r] and "length" in self.roads[r]:
                    self.length_dict[self.roads[r]["location"]] = int(self.roads[r]["length"])
            return True
        return False

    def _update_intersection_data(self, env):
        """
        Extract and update intersection data from the environment
        
        Args:
            env: Environment instance, which should have intersection and roads data
        """
            
        print(f"\n===== UPDATING INTERSECTION DATA =====")
        print(f"Environment type: {type(env).__name__}")
        if hasattr(env, '__dict__'):
            env_attrs = [attr for attr in dir(env) if not attr.startswith('__')]
            print(f"Environment attributes: {', '.join(env_attrs[:15])}")
        print("=======================================\n")
        
        
        # Store data for all intersections
        self.intersections = []
        self.roads_list = []
        self.inter_names = []
        
        # The environment from OneLine is a CityFlowEnv instance that should have
        # intersection_dict and list_intersection attributes
        if hasattr(env, 'intersection_dict') and hasattr(env, 'list_intersection') and len(env.list_intersection) > 0:
            print("Found CityFlowEnv style environment with intersection_dict")
            
            # Process each intersection
            for i in range(len(env.list_intersection)):
                intersection_idx = i
                if hasattr(env, 'current_intersection_idx'):
                    intersection_idx = env.current_intersection_idx
                
                # Get the intersection name
                inter_name = env.list_intersection[intersection_idx].inter_name
                print(f"Using intersection: {inter_name}")
                
                # Get intersection data and store it
                intersection = env.intersection_dict[inter_name]
                self.intersections.append(intersection)
                self.roads_list.append(copy.deepcopy(intersection["roads"]))
                self.inter_names.append(inter_name)
                
                # Maintain backward compatibility for every intersection
                self.intersection = intersection
                self.roads = copy.deepcopy(intersection["roads"])
                self.inter_name = inter_name
                self._update_length_dict()
            
            print(f"Successfully updated intersection data for {len(self.intersections)} intersections")
            return True
            
        # If we couldn't find intersection data, create default roads data
        print("Could not find valid intersection data, creating default roads data")
        self.roads = {
            "road_north": {"length": 300, "location": "North"},
            "road_south": {"length": 300, "location": "South"},
            "road_east": {"length": 300, "location": "East"},
            "road_west": {"length": 300, "location": "West"}
        }
        self._update_length_dict()
        print("Created default roads data")
        return False
        
    def _update_length_dict(self):
        """
        Update the length dictionary from roads data
        """
        # Initialize length dictionary
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        
        # Update from roads data
        if self.roads:
            for r in self.roads:
                if "location" in self.roads[r] and "length" in self.roads[r]:
                    location = self.roads[r]["location"]
                    if location in self.length_dict:
                        self.length_dict[location] = int(self.roads[r]["length"])