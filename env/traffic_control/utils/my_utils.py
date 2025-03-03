from typing import Dict, List, Any
import numpy as np

def get_state_detail(state: Dict, intersection_id: int) -> Dict:
    """
    Extract detailed state information for a specific intersection.
    
    Args:
        state: The full state dictionary
        intersection_id: The ID of the intersection
        
    Returns:
        Dict: Detailed state information for the intersection
    """
    if intersection_id not in state:
        return {}
    
    intersection_state = state[intersection_id]
    
    # Extract basic metrics
    current_phase = intersection_state.get('current_phase', 0)
    lane_vehicle_count = intersection_state.get('lane_vehicle_count', [])
    lane_waiting_vehicle_count = intersection_state.get('lane_waiting_vehicle_count', [])
    pressure = intersection_state.get('pressure', 0.0)
    
    # Calculate additional metrics
    total_vehicles = sum(lane_vehicle_count) if lane_vehicle_count else 0
    total_waiting = sum(lane_waiting_vehicle_count) if lane_waiting_vehicle_count else 0
    waiting_ratio = total_waiting / max(1, total_vehicles)
    
    # Identify congested lanes (more than 5 vehicles)
    congested_lanes = [i for i, count in enumerate(lane_vehicle_count) if count > 5]
    
    # Identify lanes with waiting vehicles
    waiting_lanes = [i for i, count in enumerate(lane_waiting_vehicle_count) if count > 0]
    
    # Prepare detailed state
    detail = {
        'current_phase': current_phase,
        'total_vehicles': total_vehicles,
        'total_waiting': total_waiting,
        'waiting_ratio': waiting_ratio,
        'pressure': pressure,
        'congested_lanes': congested_lanes,
        'waiting_lanes': waiting_lanes,
        'lane_vehicle_count': lane_vehicle_count,
        'lane_waiting_vehicle_count': lane_waiting_vehicle_count
    }
    
    return detail

def state2text(state: Dict, phase_names: List[str] = None) -> str:
    """
    Convert state dictionary to human-readable text.
    
    Args:
        state: The state dictionary
        phase_names: List of phase names
        
    Returns:
        str: Human-readable text representation of the state
    """
    if not state:
        return "No state information available."
    
    # Default phase names if not provided
    if phase_names is None:
        phase_names = ["WSES", "NSSS", "WLEL", "NLSL"]
    
    text = "Traffic Control Environment\n"
    text += "==========================\n\n"
    
    # Process each intersection
    for intersection_id in sorted(state.keys()):
        intersection_state = state[intersection_id]
        
        text += f"Intersection {intersection_id + 1}:\n"
        text += "-----------------\n"
        
        # Current phase
        if 'current_phase' in intersection_state:
            phase_idx = intersection_state['current_phase']
            phase_name = phase_names[phase_idx] if phase_idx < len(phase_names) else f"Phase {phase_idx}"
            text += f"Current Phase: {phase_name}\n"
        
        # Traffic metrics
        if 'pressure' in intersection_state:
            text += f"Traffic Pressure: {intersection_state['pressure']:.2f}\n"
        
        # Vehicle counts
        if 'lane_vehicle_count' in intersection_state:
            total_vehicles = sum(intersection_state['lane_vehicle_count'])
            text += f"Total Vehicles: {total_vehicles}\n"
        
        # Waiting vehicles
        if 'lane_waiting_vehicle_count' in intersection_state:
            total_waiting = sum(intersection_state['lane_waiting_vehicle_count'])
            text += f"Total Waiting: {total_waiting}\n"
            
            if total_vehicles > 0:
                waiting_ratio = total_waiting / total_vehicles
                text += f"Waiting Ratio: {waiting_ratio:.2%}\n"
        
        # Lane details
        if 'lane_vehicle_count' in intersection_state and 'lane_waiting_vehicle_count' in intersection_state:
            text += "\nLane Details:\n"
            
            # Directions for better readability
            directions = ["North", "East", "South", "West"]
            lanes_per_direction = len(intersection_state['lane_vehicle_count']) // 4
            
            for i, direction in enumerate(directions):
                text += f"  {direction}:\n"
                start_idx = i * lanes_per_direction
                end_idx = start_idx + lanes_per_direction
                
                for j in range(start_idx, end_idx):
                    if j < len(intersection_state['lane_vehicle_count']):
                        vehicles = intersection_state['lane_vehicle_count'][j]
                        waiting = intersection_state['lane_waiting_vehicle_count'][j]
                        text += f"    Lane {j+1}: {vehicles} vehicles ({waiting} waiting)\n"
        
        text += "\n"
    
    return text 