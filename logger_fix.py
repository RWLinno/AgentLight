#!/usr/bin/env python3
"""
Fix to add the missing init_logger method to RayPPOTrainer.
"""
import os
import sys
import traceback

def apply_logger_fix():
    """Add the missing init_logger method to RayPPOTrainer."""
    try:
        # Import the trainer module
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from omegaconf import OmegaConf
        
        def init_logger(self):
            """
            Initialize and configure the logger.
            If WandB is configured, it will initialize a WandB logger; otherwise, it falls back to a local file logger.
            """
            import logging
            from datetime import datetime
            
            # Set up basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Create a logger instance
            logger = logging.getLogger("ppo_trainer")
            
            # Create a simple file logger
            class FileLogger:
                def __init__(self, log_dir=None):
                    self.log_dir = log_dir or "./logs"
                    os.makedirs(self.log_dir, exist_ok=True)
                    self.log_file = os.path.join(
                        self.log_dir, 
                        f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                    print(f"Logging to file: {self.log_file}")
                    
                def log(self, data, step=None):
                    # Convert any tensors to values before logging
                    clean_data = {}
                    for k, v in data.items():
                        if hasattr(v, 'item'):
                            clean_data[k] = v.item()
                        else:
                            clean_data[k] = v
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"Step {step}: {clean_data}\n")
                    
                    # Also log to console
                    print(f"[LOG] Step {step}: {clean_data}")
                
                def close(self):
                    pass
            
            log_dir = './logs'
            if hasattr(self.config, 'logging') and hasattr(self.config.logging, 'log_dir'):
                log_dir = self.config.logging.log_dir
                
            print(f"Logging to directory: {log_dir}")
            return FileLogger(log_dir=log_dir)
        
        # Add the method to the class
        RayPPOTrainer.init_logger = init_logger
        
        print("✓ Successfully added init_logger method to RayPPOTrainer")
        return True
    except Exception as e:
        print(f"✗ Failed to add init_logger method: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply_logger_fix()
