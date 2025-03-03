from .sokoban import SokobanEnv
from .frozen_lake import FrozenLakeEnv
from .bandit import BanditEnv, TwoArmedBanditEnv
from .countdown import CountdownEnv
from .traffic_control import TrafficControlEnv
from .base import BaseEnv

ENV_REGISTRY = {
    "sokoban": SokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    "bandit": BanditEnv,
    "two_armed_bandit": TwoArmedBanditEnv,
    "countdown": CountdownEnv,
    "traffic_control": TrafficControlEnv,
}

__all__ = ['FrozenLakeEnv', 'SokobanEnv', 'BanditEnv', 'TwoArmedBanditEnv', 'CountdownEnv', 'BaseEnv', 'TrafficControlEnv']