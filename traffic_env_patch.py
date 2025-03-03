# Patch to properly handle traffic environment
import torch
import sys
import os

def patch_traffic_env():
    """
    Add necessary methods to TrafficControlEnv to handle evaluation in the training loop.
    """
    try:
        from ragen.env import TrafficControlEnv
        from verl import DataProto
        
        # Add methods required for validation
        if not hasattr(TrafficControlEnv, 'get_reward'):
            def get_reward(self):
                """Get current reward from environment."""
                if hasattr(self, 'rewards') and self.rewards is not None:
                    return float(self.rewards)
                return 0.0
            TrafficControlEnv.get_reward = get_reward
        
        if not hasattr(TrafficControlEnv, 'get_step_count'):
            def get_step_count(self):
                """Get current step count from environment."""
                if hasattr(self, 'step_count'):
                    return int(self.step_count)
                return 0
            TrafficControlEnv.get_step_count = get_step_count
            
        # Inspect the DataProto class to understand compatible types
        print("DataProto supported types:", [t.__name__ for t in DataProto._supported_types])
        
        print("Successfully patched TrafficControlEnv for training loop compatibility")
        return True
    except Exception as e:
        print(f"Error patching TrafficControlEnv: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply traffic environment patches
patch_traffic_env()
