import os
import logging
import json
import numpy as np
import random
from typing import Dict, List, Any, Tuple

class CityFlowEnv:
    """
    CityFlow environment for traffic signal control.
    
    This class provides an interface to the CityFlow traffic simulator.
    It handles the simulation state, actions, and rewards.
    """
    
    def __init__(self, path_to_log: str, path_to_work_directory: str, 
                 dic_traffic_env_conf: Dict, dic_path: Dict):
        """
        Initialize the CityFlow environment.
        
        Args:
            path_to_log: Path to log directory
            path_to_work_directory: Path to work directory containing config files
            dic_traffic_env_conf: Traffic environment configuration dictionary
            dic_path: Dictionary of paths for various files
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        
        # Extract important configuration
        self.num_intersections = self.dic_traffic_env_conf.get('NUM_INTERSECTIONS', 4)
        self.num_phases = len(self.dic_traffic_env_conf.get('PHASE', ["WSES", "NSSS", "WLEL", "NLSL"]))
        self.num_lanes = self.dic_traffic_env_conf.get('NUM_LANES', [3, 3, 3, 3])
        self.yellow_time = self.dic_traffic_env_conf.get('YELLOW_TIME', 3)
        self.min_action_time = self.dic_traffic_env_conf.get('MIN_ACTION_TIME', 10)
        
        # Initialize intersection dictionary
        self.intersection_dict = self._build_intersection_dict()
        
        # Initialize traffic state
        self.current_state = self._init_state()
        
        # Try to import CityFlow
        try:
            import cityflow
            config_file = os.path.join(self.path_to_work_directory, "config.json")
            
            # Create config file if it doesn't exist
            if not os.path.exists(config_file):
                self._create_config_file(config_file)
                
            self.eng = cityflow.Engine(config_file, thread_num=1)
            self.use_mock = False
            logging.info("Using real CityFlow engine")
        except ImportError:
            logging.warning("Using mock CityFlow engine. This is for testing only!")
            self.use_mock = True
            self._setup_mock_engine()
    
    def _create_config_file(self, config_file: str):
        """Create a CityFlow config file if it doesn't exist."""
        roadnet_file = os.path.join(self.path_to_work_directory, 
                                   self.dic_traffic_env_conf.get('ROADNET_FILE', 'roadnet.json'))
        flow_file = os.path.join(self.path_to_work_directory, 
                                self.dic_traffic_env_conf.get('TRAFFIC_FILE', 'flow.json'))
        
        config = {
            "interval": self.dic_traffic_env_conf.get('INTERVAL', 1.0),
            "seed": 0,
            "dir": self.path_to_work_directory,
            "roadnetFile": os.path.basename(roadnet_file),
            "flowFile": os.path.basename(flow_file),
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "roadnet.json",
            "replayLogFile": "replay.txt"
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _build_intersection_dict(self) -> Dict:
        """Build a dictionary of intersection information."""
        intersection_dict = {}
        
        # For a grid network
        rows = int(np.sqrt(self.num_intersections))
        cols = self.num_intersections // rows
        
        for i in range(rows):
            for j in range(cols):
                intersection_id = f"intersection_{i}_{j}"
                intersection_dict[intersection_id] = {
                    "id": intersection_id,
                    "point": {"x": i, "y": j},
                    "roads": [f"road_{i}_{j}_{k}" for k in range(4)],
                    "phases": self.dic_traffic_env_conf.get('PHASE', ["WSES", "NSSS", "WLEL", "NLSL"])
                }
        
        return intersection_dict
    
    def _setup_mock_engine(self):
        """Set up a mock engine for testing when CityFlow is not available."""
        # Load road network if available
        roadnet_file = os.path.join(self.path_to_work_directory, 
                                   self.dic_traffic_env_conf.get('ROADNET_FILE', 'roadnet.json'))
        
        # Try to load real roadnet if file exists
        if os.path.exists(roadnet_file):
            try:
                with open(roadnet_file, 'r') as f:
                    self.roadnet = json.load(f)
                    logging.info(f"Loaded roadnet from {roadnet_file}")
            except Exception as e:
                logging.warning(f"Failed to load roadnet: {e}")
                self.roadnet = self._generate_mock_roadnet()
        else:
            logging.warning(f"Roadnet file {roadnet_file} not found, using mock roadnet")
            self.roadnet = self._generate_mock_roadnet()
            
        # Initialize lane vehicle counts with random values
        self._update_mock_traffic()
    
    def _generate_mock_roadnet(self) -> Dict:
        """Generate a mock road network for testing."""
        # Simple grid network
        mock_roadnet = {
            "intersections": [],
            "roads": []
        }
        
        # Create intersections in a grid
        rows = int(np.sqrt(self.num_intersections))
        cols = self.num_intersections // rows
        
        for i in range(rows):
            for j in range(cols):
                intersection_id = f"intersection_{i}_{j}"
                roads = [f"road_{i}_{j}_{k}" for k in range(4)]
                
                intersection = {
                    "id": intersection_id,
                    "point": {"x": i * 500, "y": j * 500},
                    "roads": roads,
                    "trafficLight": {
                        "lightphases": [
                            {"availableRoadLinks": [0, 4], "time": 30},
                            {"availableRoadLinks": [1, 5], "time": 30},
                            {"availableRoadLinks": [2, 6], "time": 30},
                            {"availableRoadLinks": [3, 7], "time": 30}
                        ]
                    }
                }
                
                mock_roadnet["intersections"].append(intersection)
                
                # Create roads for this intersection
                for k, direction in enumerate(["east", "north", "west", "south"]):
                    road = {
                        "id": roads[k],
                        "startIntersection": intersection_id,
                        "endIntersection": f"intersection_{i + (0 if direction != 'east' else 1)}_{j + (0 if direction != 'north' else 1)}",
                        "lanes": self.num_lanes[k]
                    }
                    mock_roadnet["roads"].append(road)
        
        return mock_roadnet
    
    def _update_mock_traffic(self):
        """Update mock traffic state with random values."""
        self.mock_lane_vehicle_count = {}
        self.mock_lane_waiting_vehicle_count = {}
        self.mock_lane_vehicles = {}
        
        # Generate random traffic for each intersection
        for i in range(self.num_intersections):
            intersection_id = f"intersection_{i // 2}_{i % 2}"
            
            # Generate random vehicle counts for each lane
            lane_counts = np.random.randint(0, 10, size=sum(self.num_lanes))
            waiting_counts = np.random.randint(0, min(5, lane_counts.max()), size=sum(self.num_lanes))
            
            # Store in mock data structures
            self.mock_lane_vehicle_count[intersection_id] = lane_counts
            self.mock_lane_waiting_vehicle_count[intersection_id] = waiting_counts
            
            # Generate vehicle IDs
            self.mock_lane_vehicles[intersection_id] = []
            for lane_idx, count in enumerate(lane_counts):
                lane_vehicles = [f"veh_{i}_{lane_idx}_{j}" for j in range(count)]
                self.mock_lane_vehicles[intersection_id].append(lane_vehicles)
    
    def _init_state(self) -> Dict:
        """Initialize the traffic state."""
        state = {}
        
        for i in range(self.num_intersections):
            intersection_id = i
            
            # Initialize with empty/zero values
            state[intersection_id] = {
                'current_phase': 0,
                'lane_vehicle_count': [0] * sum(self.num_lanes),
                'lane_waiting_vehicle_count': [0] * sum(self.num_lanes),
                'lane_vehicles': [[] for _ in range(sum(self.num_lanes))],
                'pressure': 0.0,
                'lane_queue': [0] * sum(self.num_lanes)
            }
        
        return state
    
    def reset(self) -> Dict:
        """Reset the environment."""
        if not self.use_mock:
            self.eng.reset()
        else:
            self._update_mock_traffic()
        
        # Reset state
        self.current_state = self._init_state()
        
        # Update state with initial traffic
        self._update_state()
        
        return self.current_state
    
    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: List of actions for each intersection
            
        Returns:
            state: New state
            rewards: List of rewards for each intersection
            done: Whether the episode is done
            info: Additional information
        """
        if not self.use_mock:
            # Set traffic light phase for each intersection
            for i, action in enumerate(actions):
                self.eng.set_tl_phase(f"intersection_{i // 2}_{i % 2}", action)
            
            # Run simulation for MIN_ACTION_TIME steps
            for _ in range(self.min_action_time):
                self.eng.next_step()
        else:
            # Update mock traffic
            self._update_mock_traffic()
        
        # Update state
        self._update_state()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Always return done=False, let the environment wrapper decide when to end
        done = False
        
        # Additional info
        info = {}
        
        return self.current_state, rewards, done, info
    
    def _update_state(self):
        """Update the current state based on the simulator."""
        for i in range(self.num_intersections):
            intersection_id = i
            intersection_name = f"intersection_{i // 2}_{i % 2}"
            
            if not self.use_mock:
                # Get real data from CityFlow
                phase = self.eng.get_intersection_current_phase(intersection_name)
                
                # Get lane vehicle counts
                lane_vehicle_count = []
                lane_waiting_vehicle_count = []
                lane_vehicles = []
                
                for road in self.intersection_dict[intersection_name]["roads"]:
                    for lane_idx in range(self.num_lanes[0]):  # Assuming same number of lanes for all roads
                        lane_id = f"{road}_{lane_idx}"
                        
                        # Get vehicles on this lane
                        vehicles = self.eng.get_lane_vehicles(lane_id)
                        lane_vehicles.append(vehicles)
                        lane_vehicle_count.append(len(vehicles))
                        
                        # Get waiting vehicles (speed < 0.1 m/s)
                        waiting_vehicles = [v for v in vehicles if self.eng.get_vehicle_speed(v) < 0.1]
                        lane_waiting_vehicle_count.append(len(waiting_vehicles))
                
                # Calculate pressure (imbalance between incoming and outgoing lanes)
                incoming_lanes_count = sum(lane_vehicle_count[:len(lane_vehicle_count)//2])
                outgoing_lanes_count = sum(lane_vehicle_count[len(lane_vehicle_count)//2:])
                pressure = incoming_lanes_count - outgoing_lanes_count
                
                # Calculate queue length for each lane
                lane_queue = lane_waiting_vehicle_count
            else:
                # Use mock data
                phase = 0  # Default phase
                lane_vehicle_count = self.mock_lane_vehicle_count[intersection_name]
                lane_waiting_vehicle_count = self.mock_lane_waiting_vehicle_count[intersection_name]
                lane_vehicles = self.mock_lane_vehicles[intersection_name]
                
                # Calculate pressure
                incoming_lanes_count = sum(lane_vehicle_count[:len(lane_vehicle_count)//2])
                outgoing_lanes_count = sum(lane_vehicle_count[len(lane_vehicle_count)//2:])
                pressure = incoming_lanes_count - outgoing_lanes_count
                
                # Queue length is the same as waiting vehicle count
                lane_queue = lane_waiting_vehicle_count
            
            # Update state for this intersection
            self.current_state[intersection_id] = {
                'current_phase': phase,
                'lane_vehicle_count': lane_vehicle_count,
                'lane_waiting_vehicle_count': lane_waiting_vehicle_count,
                'lane_vehicles': lane_vehicles,
                'pressure': pressure,
                'lane_queue': lane_queue
            }
    
    def _calculate_rewards(self) -> List[float]:
        """Calculate rewards for each intersection."""
        rewards = []
        
        for i in range(self.num_intersections):
            intersection_id = i
            
            if intersection_id in self.current_state:
                # Extract metrics
                pressure = abs(self.current_state[intersection_id]['pressure'])
                queue_length = sum(self.current_state[intersection_id]['lane_queue'])
                
                # Calculate reward components based on configuration
                reward_info = self.dic_traffic_env_conf.get('DIC_REWARD_INFO', {
                    'pressure': -0.25,
                    'queue_length': -0.25
                })
                
                pressure_reward = reward_info.get('pressure', -0.25) * pressure
                queue_reward = reward_info.get('queue_length', -0.25) * queue_length
                
                # Total reward
                total_reward = pressure_reward + queue_reward
                rewards.append(total_reward)
            else:
                rewards.append(0.0)
        
        return rewards
    
    def get_state(self) -> Dict:
        """Get the current state."""
        return self.current_state
    
    def set_state(self, state: Dict):
        """Set the current state (for testing/debugging)."""
        self.current_state = state 