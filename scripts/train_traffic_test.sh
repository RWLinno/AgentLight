#!/bin/bash
# scripts/train_traffic.sh

echo "Starting training with traffic_control environment..."

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/weilinruan/AgentLight

# Create necessary directories
mkdir -p ./data/traffic

# Create minimal mock roadnet.json
echo '{
  "intersections": [
    {
      "id": "intersection_0_0",
      "point": {"x": 0, "y": 0},
      "roads": ["road_0_0_0", "road_0_0_1", "road_0_0_2", "road_0_0_3"],
      "trafficLight": {"lightphases": [
        {"availableRoadLinks": [0, 4], "time": 30},
        {"availableRoadLinks": [1, 5], "time": 30},
        {"availableRoadLinks": [2, 6], "time": 30},
        {"availableRoadLinks": [3, 7], "time": 30}
      ]}
    }
  ],
  "roads": [
    {"id": "road_0_0_0", "startIntersection": "intersection_0_0", "endIntersection": "intersection_1_0", "lanes": 3},
    {"id": "road_0_0_1", "startIntersection": "intersection_0_0", "endIntersection": "intersection_0_1", "lanes": 3},
    {"id": "road_0_0_2", "startIntersection": "intersection_0_0", "endIntersection": "intersection_-1_0", "lanes": 3},
    {"id": "road_0_0_3", "startIntersection": "intersection_0_0", "endIntersection": "intersection_0_-1", "lanes": 3}
  ]
}' > ./data/traffic/roadnet.json

# Create minimal mock traffic file
echo '{
  "vehicles": [
    {"vehicle_id": "flow_0_0", "route": ["road_0_0_0", "road_0_0_1"], "start_time": 0, "end_time": 100}
  ]
}' > ./data/traffic/anon_4_4_hangzhou_real.json

# Run with absolute minimal configuration
python verl/trainer/main_ppo.py \
    env=traffic_control \
    env.env_kwargs.path_to_work_directory=/home/weilinruan/AgentLight/data/traffic \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    data.train_files="" \
    data.val_files="" \
    trainer.total_epochs=1 \
    trainer.logger=console