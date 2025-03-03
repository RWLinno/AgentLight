# AgentLight

Note-rwl: 可解释 可泛化 可推理 通用的 交通信号控制 LLM Agent.

### data
./data/traffic

### TrafficControlEnv

```bash
bash train_traffic_final.sh
```

### TrafficControlAgent

python ./scripts/test_traffic_env.py #完善过的交通控制环境测试脚本，测试了action 0-4是否都能正常运行，模拟了action 0-4在cityflow中对应的observation

bash ./scripts/train_traffic_final.sh # 最终训练版本，跳过dataloader和data-driven valiate, 直接用cityflow的traffic数据

bash ./scripts/train_traffic_debug.sh # 调试脚本,主要测试traffic env在ray上是否正常运行，可以打印出traffic env的输出
