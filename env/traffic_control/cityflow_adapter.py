import os
import logging
import json
import numpy as np
import random
from typing import Dict, List, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cityflow_test')
logger.info('Logging system initialized for CityFlowAdapter tests')
logger.info('===========================================')
logger.info('Starting CityFlowAdapter test suite')
logger.info('This test suite validates the functionality of the CityFlow adapter')
logger.info('===========================================')

class CityFlowAdapter:
    """
    Adapter for the CityFlow traffic simulator.
    
    This class provides an interface between the RAGEN environment and the CityFlow simulator.
    If CityFlow is not available, it will use a mock implementation for testing.
    """
    
    def __init__(self, path_to_log='./log', path_to_work_directory='./data/traffic', 
                 dic_traffic_env_conf=None, dic_path=None):
        """
        Initialize the CityFlow adapter.
        
        Args:
            path_to_log (str): Path to log directory
            path_to_work_directory (str): Path to work directory
            dic_traffic_env_conf (dict): Traffic environment configuration
            dic_path (dict): Dictionary containing paths
        """
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf or {}
        self.dic_path = dic_path or {}
        
        # Extract important configuration
        self.num_intersections = self.dic_traffic_env_conf.get('NUM_INTERSECTIONS', 4)
        self.num_phases = len(self.dic_traffic_env_conf.get('PHASE', ["WSES", "NSSS", "WLEL", "NLSL"]))
        self.num_lanes = self.dic_traffic_env_conf.get('NUM_LANES', [3, 3, 3, 3])
        
        # Initialize traffic state
        self.current_state = self._init_state()
        
        # Try to import CityFlow
        try:
            import cityflow
            self.eng = cityflow.Engine(
                os.path.join(self.path_to_work_directory, "config.json"),
                thread_num=1
            )
            self.use_mock = False
            logger.info("Using real CityFlow engine")
        except ImportError:
            logger.warning("Using mock CityFlow engine. This is for testing only!")
            self.use_mock = True
            self._setup_mock_engine()

    def _setup_mock_engine(self):
        """Set up a mock engine for testing when CityFlow is not available"""
        # Load road network if available
        roadnet_file = os.path.join(self.path_to_work_directory, 
                                    self.dic_traffic_env_conf.get('ROADNET_FILE', 'roadnet.json'))
        
        # Try to load real roadnet if file exists
        if os.path.exists(roadnet_file):
            try:
                with open(roadnet_file, 'r') as f:
                    self.roadnet = json.load(f)
                    logger.info(f"Loaded roadnet from {roadnet_file}")
            except Exception as e:
                logger.warning(f"Failed to load roadnet: {e}")
                self.roadnet = self._generate_mock_roadnet()
        else:
            logger.warning(f"Roadnet file {roadnet_file} not found, using mock roadnet")
            self.roadnet = self._generate_mock_roadnet()
            
        # Initialize lane vehicle counts with random values
        self._update_mock_traffic()

    def _generate_mock_roadnet(self):
        """Generate a mock road network for testing"""
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
                roads = []
                
                # Add incoming/outgoing roads for this intersection
                for direction in ["N", "E", "S", "W"]:
                    road_id = f"road_{i}_{j}_{direction}"
                    roads.append(road_id)
                    
                mock_roadnet["intersections"].append({
                    "id": intersection_id,
                    "point": {"x": i * 500, "y": j * 500},
                    "roads": roads,
                    "trafficLight": {
                        "lightphases": [
                            {"availableRoadLinks": [0, 1], "time": 30},
                            {"availableRoadLinks": [2, 3], "time": 30},
                            {"availableRoadLinks": [4, 5], "time": 30},
                            {"availableRoadLinks": [6, 7], "time": 30}
                        ]
                    }
                })
                
                # Create roads for this intersection
                for direction, target in [
                    ("N", (i-1, j)), 
                    ("E", (i, j+1)), 
                    ("S", (i+1, j)), 
                    ("W", (i, j-1))
                ]:
                    if 0 <= target[0] < rows and 0 <= target[1] < cols:
                        target_id = f"intersection_{target[0]}_{target[1]}"
                    else:
                        target_id = "outside"
                        
                    mock_roadnet["roads"].append({
                        "id": f"road_{i}_{j}_{direction}",
                        "startIntersection": intersection_id,
                        "endIntersection": target_id,
                        "lanes": self.num_lanes[0]  # Using the first element of num_lanes
                    })
                    
        return mock_roadnet

    def _update_mock_traffic(self):
        """Update the mock traffic state with random values"""
        # For each intersection and lane, generate random vehicle counts
        for i in range(self.num_intersections):
            # Update phase (this would normally come from the real engine)
            self.current_state[i] = {
                'current_phase': random.randint(0, self.num_phases - 1),
                'lane_vehicle_count': [],
                'lane_waiting_vehicle_count': [],
                'pressure': 0.0,
                'throughput': random.randint(0, 5)
            }
            
            # Update lane vehicle counts
            total_pressure = 0
            total_lanes = sum(self.num_lanes)
            
            for lane in range(total_lanes):
                # Random number of vehicles (more realistic values)
                num_vehicles = random.randint(0, 15)
                num_waiting = random.randint(0, min(5, num_vehicles))
                
                self.current_state[i]['lane_vehicle_count'].append(num_vehicles)
                self.current_state[i]['lane_waiting_vehicle_count'].append(num_waiting)
                
                # Calculate pressure (difference between incoming and outgoing)
                lane_pressure = num_vehicles - random.randint(0, 10)
                total_pressure += abs(lane_pressure)
            
            # Update intersection pressure
            self.current_state[i]['pressure'] = total_pressure / total_lanes

    def _init_state(self) -> Dict:
        """Initialize the traffic state dictionary"""
        state = {}
        
        # Initialize state for each intersection
        for i in range(self.num_intersections):
            # Create a state dict for each intersection
            state[i] = {
                'current_phase': 0,
                'lane_vehicle_count': [0] * sum(self.num_lanes),
                'lane_waiting_vehicle_count': [0] * sum(self.num_lanes),
                'pressure': 0.0,
                'throughput': 0
            }
            
        return state

    def reset(self):
        """Reset the traffic environment"""
        if not self.use_mock:
            self.eng.reset()
        
        # Reset the current state
        self.current_state = self._init_state()
        
        # For mock engine, initialize with some random traffic
        if self.use_mock:
            self._update_mock_traffic()
        return self.current_state

    def step(self, action: Union[int, List[int]]):
        """
        Take a step in the traffic environment.
        
        Args:
            action: Action(s) to apply
                   For single intersection: int (0-3 for different phases)
                   For multiple intersections: list of ints, one per intersection
            
        Returns:
            float: Reward for this step
        """
        # Convert single action to list for multiple intersections
        if isinstance(action, int):
            action = [action] * self.num_intersections
        
        # Ensure action is a list of correct length
        if len(action) != self.num_intersections:
            logger.warning(f"Action length ({len(action)}) doesn't match number of intersections ({self.num_intersections})")
            # Pad or truncate actions to match intersection count
            if len(action) < self.num_intersections:
                action.extend([0] * (self.num_intersections - len(action)))
            else:
                action = action[:self.num_intersections]
        
        # Update the phase for each intersection
        for i in range(self.num_intersections):
            if i in self.current_state:
                self.current_state[i]['current_phase'] = action[i]
        
        if not self.use_mock:
            # For real CityFlow engine
            logger.debug(f"Using real CityFlow engine with actions: {action}")
            
            # Set traffic light phases for each intersection
            for i in range(self.num_intersections):
                intersection_id = f"intersection_{i//2}_{i%2}"
                logger.debug(f"Setting phase {action[i]} for {intersection_id}")
                self.eng.set_tl_phase(intersection_id, action[i])
            
            # Advance simulation
            self.eng.next_step()
            
            # Get updated state
            total_reward = 0
            for i in range(self.num_intersections):
                intersection_id = f"intersection_{i//2}_{i%2}"
                
                # Update current phase
                self.current_state[i]['current_phase'] = action[i]
                
                # Get vehicle counts
                lane_vehicle_count = []
                lane_waiting_vehicle_count = []
                total_pressure = 0
                
                # Get vehicle counts for each lane
                total_lanes = sum(self.num_lanes)
                for lane in range(total_lanes):
                    # Calculate road and lane IDs
                    direction_idx = lane // self.num_lanes[0]
                    lane_idx = lane % self.num_lanes[0]
                    directions = ["N", "E", "S", "W"]
                    direction = directions[direction_idx % 4]
                    road_id = f"road_{i//2}_{i%2}_{direction}"
                    
                    # Get counts from CityFlow
                    try:
                        vehicles = self.eng.get_lane_vehicle_count(f"{road_id}_{lane_idx}")
                        waiting = self.eng.get_lane_waiting_vehicle_count(f"{road_id}_{lane_idx}")
                    except Exception as e:
                        logger.warning(f"Error getting vehicle count: {e}, using default values")
                        vehicles = 0
                        waiting = 0
                    
                    lane_vehicle_count.append(vehicles)
                    lane_waiting_vehicle_count.append(waiting)
                    
                    # Contribute to pressure calculation
                    lane_pressure = abs(vehicles - waiting)
                    total_pressure += lane_pressure
                
                # Update state
                self.current_state[i]['lane_vehicle_count'] = lane_vehicle_count
                self.current_state[i]['lane_waiting_vehicle_count'] = lane_waiting_vehicle_count
                self.current_state[i]['pressure'] = total_pressure / total_lanes if total_lanes > 0 else 0
                
                # Calculate reward components
                reward_info = self.dic_traffic_env_conf.get('DIC_REWARD_INFO', {})
                pressure_reward = reward_info.get('pressure', -0.25) * self.current_state[i]['pressure']
                queue_reward = reward_info.get('queue_length', -0.25) * sum(lane_waiting_vehicle_count)
                
                # Calculate throughput (vehicles that successfully passed)
                throughput = self.eng.get_vehicle_count() - sum(lane_vehicle_count)
                throughput = max(0, throughput)  # Ensure throughput is non-negative
                self.current_state[i]['throughput'] = throughput
                throughput_reward = reward_info.get('throughput', 0.5) * throughput
                
                # Add to total reward
                intersection_reward = pressure_reward + queue_reward + throughput_reward
                total_reward += intersection_reward
                
                logger.debug(f"Intersection {i} - Pressure: {pressure_reward:.2f}, "
                           f"Queue: {queue_reward:.2f}, Throughput: {throughput_reward:.2f}")
        else:
            # For mock engine, simulate traffic changes
            self._update_mock_traffic()
            
            # Update current phases in the state based on actions
            for i in range(self.num_intersections):
                if i in self.current_state:
                    self.current_state[i]['current_phase'] = action[i]
            
            # Calculate reward based on pressure and queue length
            total_reward = 0
            reward_info = self.dic_traffic_env_conf.get('DIC_REWARD_INFO', {})
            
            for i in range(self.num_intersections):
                if i in self.current_state:
                    pressure = self.current_state[i]['pressure']
                    queue_length = sum(self.current_state[i]['lane_waiting_vehicle_count'])
                    throughput = self.current_state[i]['throughput']
                    
                    pressure_reward = reward_info.get('pressure', -0.25) * pressure
                    queue_reward = reward_info.get('queue_length', -0.25) * queue_length
                    throughput_reward = reward_info.get('throughput', 0.5) * throughput
                    
                    intersection_reward = pressure_reward + queue_reward + throughput_reward
                    total_reward += intersection_reward
                    
                    logger.debug(f"Mock Intersection {i} - Pressure: {pressure_reward:.2f}, "
                               f"Queue: {queue_reward:.2f}, Throughput: {throughput_reward:.2f}")
        
        return total_reward

    def get_state(self):
        """
        Get the current traffic state.
        
        Returns:
            dict: Current traffic state
        """
        return self.current_state.copy()

    def set_state(self, state):
        """
        Set the traffic state (for copying environments).
        
        Args:
            state (dict): Traffic state to set
        """
        self.current_state = state.copy() 