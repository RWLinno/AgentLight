import os
import json
import shutil
import argparse
import urllib.request

def download_file(url, destination):
    """Download file from URL to destination."""
    print(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded {destination}")

def main(args):
    """Prepare traffic data files."""
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Base URLs for traffic data files
    base_url = "https://raw.githubusercontent.com/usail-hkust/LLMTSCS/main/data"
    
    # Download roadnet file if it doesn't exist
    roadnet_file = os.path.join(args.data_dir, "roadnet.json")
    if not os.path.exists(roadnet_file) or args.force:
        download_file(f"{base_url}/roadnet_4_4.json", roadnet_file)
    
    # Download traffic file if it doesn't exist
    traffic_file = os.path.join(args.data_dir, args.traffic_file)
    if not os.path.exists(traffic_file) or args.force:
        download_file(f"{base_url}/{args.traffic_file}", traffic_file)
    
    print(f"Traffic data prepared in {args.data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare traffic data files")
    parser.add_argument("--data_dir", type=str, default="./data/traffic", 
                        help="Directory to store traffic data")
    parser.add_argument("--traffic_file", type=str, default="anon_4_4_hangzhou_real.json",
                        help="Traffic flow file to download")
    parser.add_argument("--force", action="store_true", 
                        help="Force download even if files exist")
    
    args = parser.parse_args()
    main(args) 