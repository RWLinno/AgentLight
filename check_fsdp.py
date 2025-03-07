import torch
import os
import sys

def check_fsdp_support():
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. FSDP requires CUDA.")
        return False
        
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Check FSDP import
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        print("FSDP can be imported successfully")
    except ImportError:
        print("FSDP not available in your PyTorch version")
        return False
    
    # Try to initialize process group
    try:
        if not torch.distributed.is_initialized():
            # Single GPU test - using gloo as it doesn't require multiple GPUs
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            torch.distributed.init_process_group(backend="gloo")
            print("Process group initialized successfully")
            
            # Clean up
            torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"Error initializing process group: {e}")
        return False
    
    # Simple FSDP test with a tiny model
    try:
        # Create a very simple model
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo")
        
        model = torch.nn.Linear(10, 10).cuda()
        wrapped_model = FSDP(model)
        print("Successfully created FSDP model")
        
        # Clean up
        torch.distributed.destroy_process_group()
        return True
    except Exception as e:
        print(f"Error in FSDP test: {e}")
        return False

if __name__ == "__main__":
    if check_fsdp_support():
        print("\n✅ Your system supports FSDP!")
    else:
        print("\n❌ Your system has issues with FSDP support.")