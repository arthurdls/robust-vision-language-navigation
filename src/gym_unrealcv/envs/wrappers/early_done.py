import gymnasium as gym
from gymnasium import Wrapper
import time

class EarlyDoneWrapper(Wrapper):
    def __init__(self, env, max_lost_steps=50):
        super().__init__(env)
        self.max_lost_steps = max_lost_steps

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if  not info['metrics']['target_viewed']:
            self.count_lost += 1
        else:
            self.count_lost = 0
        if self.count_lost > self.max_lost_steps:
            info['Done'] = True
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.count_lost = 0
        return obs, info
