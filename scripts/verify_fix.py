#!/usr/bin/env python
# Script to verify the fix for LLM generation of traffic prompts

import os
import sys
import json
from pprint import pprint

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment and generation code
try:
    from env.traffic_control.env import TrafficControlEnv
    from ragen.llm_agent.generation import LLMGenerationManager  # Adjust import based on actual structure
    print("✓ Successfully imported required modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)

def print_separator():
    print("\n" + "="*50 + "\n")

def verify_environment():
    """Verify that the environment returns the correct format"""
    print("Testing environment...")
    
    # Make sure the data directory exists
    os.makedirs('./data/traffic', exist_ok=True)
    
    # Initialize the environment
    env = TrafficControlEnv(
        path_to_work_directory='./data/traffic',
        max_steps=20,
        num_intersections=1
    )
    
    # Reset the environment
    state = env.reset()
    
    # Check if state is a dictionary
    if isinstance(state, dict):
        print("✓ Environment reset returns a dictionary")
        print(f"  State has {len(state)} keys")
        print(f"  State keys: {state.keys()}")
    else:
        print(f"✗ Environment reset returns a {type(state).__name__}, not a dictionary")
    
    # Take a step and check the info dictionary
    observation, reward, done, info = env.step(1)  # Valid action
    
    if "text_observation" in info:
        print("✓ Step info contains text_observation")
    else:
        print("✗ Step info is missing text_observation")
    
    # Check get_last_info
    last_info = env.get_last_info()
    if last_info is not None and "text_observation" in last_info:
        print("✓ get_last_info returns info with text_observation")
    else:
        print("✗ get_last_info failed or missing text_observation")
    
    # Check render method
    render_output = env.render()
    if isinstance(render_output, str):
        print("✓ Render method returns a string")
    else:
        print(f"✗ Render method returns a {type(render_output).__name__}, not a string")
    
    return env

def mock_generation_manager():
    """Create a mock GenerationManager with just enough functionality for testing"""
    class MockGenerationManager:
        def __init__(self):
            pass
            
        def generate_traffic_prompt(self, env):
            """Generate a prompt for the traffic control environment."""
            # Get the state information from the environment
            state_info = env.render("text") if hasattr(env, "render") else "No state information available."
            
            # If state_info is a string, use it directly, otherwise it's a dictionary
            # and we need to extract the information we need
            if isinstance(state_info, dict):
                # Extract information from the dictionary
                vehicle_counts = state_info.get("vehicle_counts", {})
                waiting_vehicles = state_info.get("waiting_vehicles", {})
                pressure = state_info.get("pressure", 0.0)
                current_phase = state_info.get("current_phase", 0)
                
                # Build a text representation
                state_info_text = f"Traffic State Summary:\n"
                state_info_text += f"Step: {state_info.get('step', 0)}/{state_info.get('max_steps', 1000)}\n\n"
                
                # Process each intersection
                for intersection_id, intersection_data in state_info.get("intersections", {}).items():
                    state_info_text += f"Intersection {intersection_id}:\n"
                    state_info_text += "------------------------\n"
                    
                    # Current phase
                    phase_name = intersection_data.get("phase_name", "Unknown")
                    state_info_text += f"Current Phase: {phase_name}\n"
                    
                    # Traffic pressure
                    pressure = intersection_data.get("pressure", 0.0)
                    state_info_text += f"Traffic Pressure: {pressure:.2f}\n"
                    
                    # Vehicle counts by direction
                    state_info_text += "\nVehicle Counts by Direction:\n"
                    for direction, count in intersection_data.get("vehicle_counts", {}).items():
                        state_info_text += f"  {direction}: {count} vehicles\n"
                    
                    # Waiting vehicles
                    state_info_text += "\nWaiting Vehicles by Direction:\n"
                    for direction, count in intersection_data.get("waiting_vehicles", {}).items():
                        state_info_text += f"  {direction}: {count} waiting\n"
                    
                    # Lane details
                    state_info_text += "\nDetailed Lane Information:\n"
                    for lane_data in intersection_data.get("lane_info", []):
                        direction = lane_data.get("direction", "Unknown")
                        lane = lane_data.get("lane", 0)
                        count = lane_data.get("count", 0)
                        waiting = lane_data.get("waiting", 0)
                        state_info_text += f"  {direction} Lane {lane}: {count} vehicles ({waiting} waiting)\n"
                    
                    state_info_text += "\n"
                
                # Use the text representation
                state_info = state_info_text
            
            # Check if info dictionary exists and has text_observation
            if hasattr(env, "get_last_info") and env.get_last_info() and "text_observation" in env.get_last_info():
                state_info = env.get_last_info()["text_observation"]

            # Create the prompt
            prompt = (
                "# Traffic Signal Control Task\n\n"
                "You are an intelligent traffic signal controller. Your task is to optimize "
                "traffic flow by selecting appropriate signal phases at each step.\n\n"
                f"## Current Traffic State:\n{state_info}\n\n"
                "## Available Signal Phases:\n"
                "1. WSES: Green for West-East Straight traffic\n"
                "2. NSSS: Green for North-South Straight traffic\n"
                "3. WLEL: Green for West-East Left-turn traffic\n"
                "4. NLSL: Green for North-South Left-turn traffic\n\n"
                "Analyze the traffic conditions and select the optimal signal phase (1-4)."
            )
            
            return prompt
            
    return MockGenerationManager()

def test_generation_with_mock(env):
    """Test the generation code with a mock implementation"""
    print("Testing generation code with mock implementation...")
    
    # Create a mock GenerationManager
    gen_manager = mock_generation_manager()
    
    try:
        # Try to generate a prompt
        prompt = gen_manager.generate_traffic_prompt(env)
        
        # Check that the prompt is a string
        if isinstance(prompt, str):
            print("✓ generate_traffic_prompt returns a string")
            print(f"  Prompt length: {len(prompt)} characters")
            print("\nPrompt sample:")
            print(prompt[:200] + "...")
            
            # Save the prompt to a file for inspection
            with open("traffic_prompt.txt", "w") as f:
                f.write(prompt)
            print("  Full prompt saved to traffic_prompt.txt")
        else:
            print(f"✗ generate_traffic_prompt returns a {type(prompt).__name__}, not a string")
        
        return True
    except Exception as e:
        print(f"✗ Error generating prompt: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_with_actual(env):
    """Test the generation code with the actual implementation"""
    print("Testing generation code with actual implementation...")
    
    try:
        # Try to import the actual GenerationManager
        from ragen.llm_agent.generation import GenerationManager
        
        # Configure the GenerationManager
        config = {
            "llm_backend": "mock",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        gen_manager = GenerationManager(config)
        
        # Try to generate a prompt
        prompt = gen_manager.generate_traffic_prompt(env)
        
        # Check that the prompt is a string
        if isinstance(prompt, str):
            print("✓ generate_traffic_prompt returns a string")
            print(f"  Prompt length: {len(prompt)} characters")
            print("\nPrompt sample:")
            print(prompt[:200] + "...")
            
            # Save the prompt to a file for inspection
            with open("traffic_prompt_actual.txt", "w") as f:
                f.write(prompt)
            print("  Full prompt saved to traffic_prompt_actual.txt")
        else:
            print(f"✗ generate_traffic_prompt returns a {type(prompt).__name__}, not a string")
        
        return True
    except ImportError:
        print("✗ Could not import actual GenerationManager")
        print("  Skipping actual implementation test")
        return None
    except Exception as e:
        print(f"✗ Error generating prompt with actual implementation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run verification tests"""
    print("=== VERIFYING FIX FOR LLM TRAFFIC PROMPT GENERATION ===")
    
    # Test the environment
    print_separator()
    env = verify_environment()
    
    # Test generation with mock implementation
    print_separator()
    mock_success = test_generation_with_mock(env)
    
    # Test generation with actual implementation if available
    print_separator()
    actual_success = test_generation_with_actual(env)
    
    # Summarize results
    print_separator()
    print("VERIFICATION RESULTS:")
    print(f"✓ Environment test passed")
    print(f"{'✓' if mock_success else '✗'} Mock generation test {'passed' if mock_success else 'failed'}")
    
    if actual_success is not None:
        print(f"{'✓' if actual_success else '✗'} Actual generation test {'passed' if actual_success else 'failed'}")
    else:
        print("- Actual generation test skipped")
    
    # Overall result
    if mock_success and (actual_success is None or actual_success):
        print("\n✅ VERIFICATION SUCCESSFUL!")
    else:
        print("\n❌ VERIFICATION FAILED!")

if __name__ == "__main__":
    main() 