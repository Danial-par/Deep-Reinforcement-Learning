import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Configurable parameters for RND + PPO training")

    parser.add_argument("--n_workers", default=2, type=int, help="Number of parallel environments.")
    parser.add_argument("--interval", default=50, type=int, help="Checkpoint/log interval.")
    parser.add_argument("--do_test", action="store_true", help="Evaluate a trained agent.")
    parser.add_argument("--render", action="store_true", help="Render environment during execution.")
    parser.add_argument("--train_from_scratch", action="store_true", help="If not set, loads previous model.")

    args = parser.parse_args()

    # === Default hyperparameters from the RND paper and tuned for MiniGrid ===
    default_params = {
        # ENVIRONMENT
        "env_name": "MiniGrid-Empty-5x5-v0",
        "state_shape": (3, 7, 7),   # MiniGrid observation shape
        "obs_shape": (3, 7, 7),

        # TRAINING
        "total_rollouts_per_env": 10000,
        "max_frames_per_episode": 200,
        "rollout_length": 128,
        "n_epochs": 4,
        "n_mini_batch": 4,
        "lr": 2.5e-4,

        # PPO
        "ext_gamma": 0.99,
        "int_gamma": 0.99,
        "lambda": 0.95,
        "clip_range": 0.1,
        "max_grad_norm": 0.5,
        "ent_coeff": 0.001,

        # Advantage weighting
        "ext_adv_coeff": 1.0,
        "int_adv_coeff": 1.0,

        # RND
        "pre_normalization_steps": 10,
        "predictor_proportion": 0.25,  # Fraction of predictor features updated each batch

        # MISC
        "seed": 123
    }

    total_params = {**vars(args), **default_params}
    print("\nConfiguration:")
    for k, v in total_params.items():
        print(f"  {k}: {v}")
    return total_params
