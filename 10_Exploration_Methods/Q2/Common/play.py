import gymnasium as gym
import numpy as np
import torch
import time
from torch import device as torch_device
from Common.utils import make_env

def extract_obs_image(obs):
    """Extracts the 'image' key from MiniGrid observation."""
    if isinstance(obs, tuple):
        obs, _ = obs
    if isinstance(obs, dict):
        return obs.get("image", obs)
    return obs

def preprocess(obs, device):
    """Convert HWC observation to normalized CHW torch tensor."""
    return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

class Play:
    def __init__(self, env_name, agent, checkpoint, max_episode=1, render=True):
        self.env = make_env(env_name, max_steps=4500)()
        self.agent = agent
        self.agent.set_from_checkpoint(checkpoint)
        self.agent.set_to_eval_mode()
        self.device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episode = max_episode
        self.render = render

    def evaluate(self):
        all_rewards = []

        try:
            for ep in range(self.max_episode):
                obs = extract_obs_image(self.env.reset(seed=ep))
                obs_tensor = preprocess(obs, self.device)
                hidden_state = torch.zeros(1, 256).to(self.device)
                episode_reward = 0
                done = False

                while not done:
                    with torch.no_grad():
                        action, *_ , hidden_state = self.agent.get_actions_and_values(obs_tensor, hidden_state)

                    obs, reward, terminated, truncated, _ = self.env.step(action.item())
                    obs = extract_obs_image(obs)
                    obs_tensor = preprocess(obs, self.device)

                    episode_reward += reward
                    done = terminated or truncated

                    if self.render:
                        self.env.render()
                        time.sleep(0.05)

                print(f"Episode {ep + 1} Reward: {episode_reward:.2f}")
                all_rewards.append(episode_reward)

            avg_reward = np.mean(all_rewards)
            print(f"\nAverage Reward over {self.max_episode} episodes: {avg_reward:.2f}")

        finally:
            self.env.close()
