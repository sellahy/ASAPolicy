idm_training_config_dict = {
    "seed" : 42,
    "epochs" : 100,
    "lr" : 1e-03,
    "batch_size" : 128,
    "training_envs" : ["chess_env/ChessWorld-v0", "chess_env/BishopWorld-v0", "chess_env/CamelWorld-v0", "chess_env/GoldGeneralWorld-v0"], # names of training environments as they would be used with gym.make
    "dataset_size" : 12800,
    "width" : 5,
    "height" : 5,
    "project_name" : "simple_idm",
    "wandb_entity" : "sellahy-university-of-maryland-at-college-park"
    }

policy_training_config_dict = {
    "wandb_entity" : "sellahy-university-of-maryland-at-college-park",
    "project_name" : "simple_policy",
    "training_envs" : ["chess_env/KingWorld-v0", "chess_env/KnightWorld-v0"],
    "jumpstart_envs" : ["chess_env/SilverGeneralWorld-v0", "chess_env/ZebraWorld-v0"],
    "width" : 5,
    "height" : 5,
    "gamma" : 0.99,
    "clip_epsilon" : 0.2,
    "lr" : 0.001,
    "epochs" : 200,
}
