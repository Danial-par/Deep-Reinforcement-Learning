# main.py
# Main training/evaluation entry point for PPO + RND agent in MiniGrid environments.

import gymnasium as gym
import minigrid
import numpy as np
import torch
from Core.ppo_rnd_agent import Brain
from Common.logger import Logger
from Common.config import get_params
from Common.utils import set_random_seeds
from Common.play import Play
from tqdm import tqdm

def extract_image(obs_dict):
    """Extracts and formats observation image as [C, H, W]."""
    image = obs_dict['image'] if isinstance(obs_dict, dict) else obs_dict[0]['image']
    return np.transpose(image, (2, 0, 1))  # HWC -> CHW

if __name__ == '__main__':
    # --- Config & Seeding ---
    config = get_params()
    set_random_seeds(config["seed"])

    # --- Get action space from env and update config ---
    temp_env = gym.make(config["env_name"])
    obs_img = extract_image(temp_env.reset())
    config["n_actions"] = temp_env.action_space.n
    temp_env.close()

    config["batch_size"] = (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]
    config["predictor_proportion"] = 32 / config["n_workers"]

    # --- Init PPO + RND agent and logger ---
    T = config["rollout_length"]
    brain = Brain(**config)
    logger = Logger(brain, **config)

    if not config["do_test"]:  # --- TRAIN ---
        if not config["train_from_scratch"]:
            checkpoint = logger.load_weights()
            brain.set_from_checkpoint(checkpoint)
            init_iteration = checkpoint["iteration"]
            episode = checkpoint["episode"]
        else:
            init_iteration = 0
            episode = 0

        env = gym.make(config["env_name"])
        logger.on()

        for iteration in tqdm(range(init_iteration + 1, config["total_rollouts_per_env"] + 1)):
            obs = torch.tensor(extract_image(env.reset()), dtype=torch.float32).unsqueeze(0)
            hidden_state = torch.zeros(1, 256)
            done = False
            ep_ext_reward, ep_int_reward = 0, 0
            rollout = []

            for t in range(T):
                with torch.no_grad():
                    action, int_val, ext_val, log_prob, probs, hidden_state = brain.get_actions_and_values(obs, hidden_state)

                next_obs_dict, ext_reward, terminated, truncated, _ = env.step(action.item())
                next_obs = torch.tensor(extract_image(next_obs_dict), dtype=torch.float32).unsqueeze(0)
                done = terminated or truncated

                int_r = brain.calculate_int_rewards(next_obs.numpy().squeeze(0), batch=False).item()
                norm_int_r = brain.normalize_int_rewards([[int_r]])[0][0]

                rollout.append((obs, action, ext_reward, norm_int_r, done, log_prob, ext_val, int_val, hidden_state))

                obs = next_obs
                ep_ext_reward += ext_reward
                ep_int_reward += norm_int_r

                if done:
                    break

            # --- Prepare rollout batch ---
            obs_batch = torch.cat([x[0] for x in rollout])
            action_batch = torch.tensor([x[1] for x in rollout])
            ext_rewards = np.array([x[2] for x in rollout])
            int_rewards = np.array([x[3] for x in rollout])
            dones = np.array([x[4] for x in rollout], dtype=np.float32)
            log_probs = torch.tensor([x[5] for x in rollout])
            ext_vals = np.array([x[6].item() for x in rollout] + [0.0])
            int_vals = np.array([x[7].item() for x in rollout] + [0.0])
            hiddens = torch.cat([x[8] for x in rollout])

            # --- GAE & Training ---
            ext_adv = brain.get_gae([ext_rewards], [ext_vals[:-1]], [ext_vals[1:]], [dones], config["ext_gamma"])
            int_adv = brain.get_gae([int_rewards], [int_vals[:-1]], [int_vals[1:]], [np.zeros_like(dones)], config["int_gamma"])
            advs = ext_adv * config["ext_adv_coeff"] + int_adv * config["int_adv_coeff"]

            training_logs = brain.train(
                states=obs_batch,
                actions=action_batch,
                int_rewards=int_rewards,
                ext_rewards=ext_rewards,
                dones=dones,
                int_values=int_vals[:-1],
                ext_values=ext_vals[:-1],
                log_probs=log_probs,
                next_int_values=int_vals[1:],
                next_ext_values=ext_vals[1:],
                total_next_obs=obs_batch.numpy(),
                hidden_states=hiddens
            )

            logger.log_iteration(iteration, training_logs, ep_int_reward, probs.max().item())
            episode += 1
            logger.log_episode(episode, ep_ext_reward, _)

        env.close()
        logger.off()

    else:  # --- EVALUATION ---
        checkpoint = logger.load_weights()
        play = Play(config["env_name"], brain, checkpoint)
        play.evaluate()
