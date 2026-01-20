import gymnasium as gym
from gym_hybrid.environments import MovingEnv
from gym_hybrid.environments import SlidingEnv

gym.register(
    id='Moving-v0',
    entry_point='gym_hybrid:MovingEnv',
)
gym.register(
    id='Sliding-v0',
    entry_point='gym_hybrid:SlidingEnv',
)