idm_training_config_dict: dict = {
    "seed":           42,
    "epochs":         100,
    "lr":             1e-03,
    "batch_size":     128,
    "all_envs" : [
        "chess_env/ChessWorld-v0",
        "chess_env/BishopWorld-v0",
        "chess_env/CamelWorld-v0",
        "chess_env/GoldGeneralWorld-v0",
        "chess_env/KingWorld-v0",
        "chess_env/KnightWorld-v0",
        "chess_env/SilverGeneralWorld-v0",
        "chess_env/ZebraWorld-v0",
    ],
    # Names of environments used for IDM training (passed to gym.make).
    # All environments listed here will have a per-environment decoder trained
    # alongside the shared IDM encoder.
    "training_envs":  [
        "chess_env/ChessWorld-v0",
        "chess_env/BishopWorld-v0",
        "chess_env/CamelWorld-v0",
        "chess_env/GoldGeneralWorld-v0",
    ],
    "dataset_size":   12800,
    "width":          5,
    "height":         5,
    "project_name":   "action_space_agnostic_agent",
    "wandb_entity":   "sellahy-university-of-maryland-at-college-park",
    "idm_latent_dim": 3,  # latent dimension of the shared IDM encoder
}

ppo_config_dict: dict = {
    # PPO training hyperparameters.
    # Must not redeclare any key already in idm_training_config_dict —
    # the assertion below will catch any accidental overlaps.
    "total_timesteps":    200_000,
    "gamma":              0.99,
    "clip_epsilon":       0.2,
    "ppo_lr":             1e-3,       # separate from IDM lr to avoid key collision
    "eval_frequency":     10,         # episodes between in-training evaluations
    "eval_episodes":      100,        # episodes per evaluation call

    # ASA policy transformer architecture
    "latent_action_dim":  3,
    "transformer_nhead":  1,
    "transformer_layers": 1,
    "dim_feedforward":    64,

    # Environments the ASA agent trains on.
    # Must be the same environments that the IDM saw. That is, it is 
    # equivalent to idm_training_config_dict["training_envs"].
    "asa_envs": idm_training_config_dict["training_envs"],

    # Metric / plotting config consumed by plot_results.py
    "jumpstart_K":           100,   # initial eval events that define jumpstart
    "asymptote_K":           100,   # final eval events that define asymptotic perf
    "convergence_threshold": 0.8,   # fraction of asymptote used for convergence speed
}

# Guard against accidental key collisions before merging.
# If a key appears in both dicts, the merge would silently overwrite the
# idm value
_overlap: set = set(idm_training_config_dict) & set(ppo_config_dict)
assert not _overlap, (
    f"ppo_config_dict redefines keys already in idm_training_config_dict: {_overlap}. "
    f"Remove the duplicate keys from ppo_config_dict."
)

# Single merged config consumed by run_experiment.py and W&B sweeps.
base_config: dict = {**idm_training_config_dict, **ppo_config_dict}
