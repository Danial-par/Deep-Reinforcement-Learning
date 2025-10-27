import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque

class Logger:
    NUM_LOG_METRICS = 7  # PG Loss, Ext/Int Loss, RND Loss, Entropy, Int/Ext EV

    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_ext_reward = 0
        self.running_ext_reward = 0
        self.running_int_reward = 0
        self.running_act_prob = 0
        self.running_training_logs = np.zeros(self.NUM_LOG_METRICS, dtype=np.float32)
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=self.moving_avg_window)
        self.running_last_10_ext_r = 0

        self.exp_avg = lambda x, y: 0.9 * x + 0.1 * y

        if not self.config["do_test"] and self.config["train_from_scratch"]:
            self.create_weights_folder()
            self.log_params()

    @property
    def log_path(self):
        return os.path.join("Logs", self.log_dir)

    @property
    def model_path(self):
        return os.path.join("Models", self.log_dir)

    def create_weights_folder(self):
        os.makedirs(self.model_path, exist_ok=True)

    def log_params(self):
        with SummaryWriter(self.log_path) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, iteration, training_logs, int_reward, action_prob):
        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)

        # Flatten training logs robustly
        flat_logs = []
        for item in training_logs:
            try:
                # Handle PyTorch tensors
                if hasattr(item, 'cpu'):  # Check if it's a tensor
                    arr = item.cpu().detach().numpy().astype(np.float32).flatten()
                else:
                    arr = np.array(item, dtype=np.float32).flatten()
                flat_logs.extend(arr.tolist())
            except Exception as e:
                print(f"[Logger Warning] Skipping non-numeric log entry: {item} ({e})")

        flat_logs = np.array(flat_logs[:self.NUM_LOG_METRICS], dtype=np.float32)
        self.running_training_logs = self.exp_avg(self.running_training_logs, flat_logs)

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        # Log scalars
        log_names = [
            "PG Loss", "Ext Value Loss", "Int Value Loss",
            "RND Loss", "Entropy", "Intrinsic EV", "Extrinsic EV"
        ]

        with SummaryWriter(self.log_path) as writer:
            writer.add_scalar("Episode Ext Reward", self.episode_ext_reward, self.episode)
            writer.add_scalar("Running Episode Ext Reward", self.running_ext_reward, self.episode)
            writer.add_scalar("Running Last 10 Ext Reward", self.running_last_10_ext_r, self.episode)
            writer.add_scalar("Max Episode Ext Reward", self.max_episode_reward, self.episode)
            writer.add_scalar("Running Action Probability", self.running_act_prob, iteration)
            writer.add_scalar("Running Intrinsic Reward", self.running_int_reward, iteration)

            for i, name in enumerate(log_names):
                writer.add_scalar(name, self.running_training_logs[i], iteration)

        self.off()
        if iteration % self.config["interval"] == 0:
            print(f"Iter: {iteration} | EP: {self.episode} | ExtR: {self.episode_ext_reward:.2f} | "
                  f"Running ExtR: {self.running_ext_reward:.2f} | Duration: {self.duration:.2f}s | "
                  f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        self.on()

    def log_episode(self, episode, episode_ext_reward, _):
        self.episode = episode
        self.episode_ext_reward = episode_ext_reward
        self.max_episode_reward = max(self.max_episode_reward, episode_ext_reward)
        self.running_ext_reward = self.exp_avg(self.running_ext_reward, episode_ext_reward)

        self.last_10_ep_rewards.append(episode_ext_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_ext_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')[0]

    def save_params(self, episode, iteration):
        torch.save({
            "current_policy_state_dict": self.brain.current_policy.state_dict(),
            "predictor_model_state_dict": self.brain.predictor_model.state_dict(),
            "target_model_state_dict": self.brain.target_model.state_dict(),
            "optimizer_state_dict": self.brain.optimizer.state_dict(),
            "state_rms_mean": self.brain.state_rms.mean,
            "state_rms_var": self.brain.state_rms.var,
            "state_rms_count": self.brain.state_rms.count,
            "int_reward_rms_mean": self.brain.int_reward_rms.mean,
            "int_reward_rms_var": self.brain.int_reward_rms.var,
            "int_reward_rms_count": self.brain.int_reward_rms.count,
            "iteration": iteration,
            "episode": episode,
            "running_reward": self.running_ext_reward,
        }, os.path.join(self.model_path, "params.pth"))

    def load_weights(self):
        model_dirs = sorted(glob.glob("Models/*"))
        if not model_dirs:
            raise FileNotFoundError("No saved model directory found.")
        checkpoint_path = os.path.join(model_dirs[-1], "params.pth")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.log_dir = os.path.basename(model_dirs[-1])
        return checkpoint
