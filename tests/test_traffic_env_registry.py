import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragen.env import ENV_REGISTRY

def test_environment_registry():
    """Test that the environment registry contains our environments."""
    print("Environment Registry:")
    for env_name, env_class in ENV_REGISTRY.items():
        print(f"  - {env_name}: {env_class.__name__}")
    
    # Check if traffic_control is in the registry
    if "traffic_control" in ENV_REGISTRY:
        print("\ntraffic_control environment is registered correctly!")
    else:
        print("\nWARNING: traffic_control environment is NOT registered!")
        
    # If traffic_control is not in the registry, try to import it directly
    if "traffic_control" not in ENV_REGISTRY:
        try:
            from traffic.traffic_adapter import TrafficControlEnv
            print(f"TrafficControlEnv can be imported directly: {TrafficControlEnv}")
            
            # Try to register it manually
            ENV_REGISTRY["traffic_control"] = TrafficControlEnv
            print("Manually registered TrafficControlEnv")
        except ImportError as e:
            print(f"Failed to import TrafficControlEnv: {e}")

if __name__ == "__main__":
    test_environment_registry() 