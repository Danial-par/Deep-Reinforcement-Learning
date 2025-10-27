from torch.multiprocessing import Process
import numpy as np
from Common.utils import make_env

class Worker(Process):
    def __init__(self, worker_id, conn, **config):
        super().__init__()
        self.id = worker_id
        self.conn = conn
        self.config = config
        self.env = make_env(config["env_name"], config["max_frames_per_episode"])()
        self.max_steps = config["max_frames_per_episode"]
        self.render_enabled = config.get("render", False)

        self.reset_env()

    def reset_env(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):  # Gym >= 0.26 compatibility
            obs = obs[0]
        self.obs = obs
        self.steps = 0

    def render(self):
        if self.render_enabled:
            self.env.render()

    def run(self):
        while True:
            self.conn.send(self.obs)
            action = self.conn.recv()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated or (self.steps >= self.max_steps)
            self.steps += 1

            if isinstance(next_obs, tuple):  # Gym >= 0.26 compatibility
                next_obs = next_obs[0]

            self.render()
            self.conn.send((next_obs, reward, done, info))

            if done:
                self.reset_env()
            else:
                self.obs = next_obs
