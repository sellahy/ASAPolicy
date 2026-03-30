from gymnasium.envs.registration import register

register(
    id="chess_env/ChessWorld-v0",
    entry_point="chess_env.envs:ChessWorldEnv",
)
# how to make:
# import gymnasium as gym
# import chess_env
# env = gym.make("chess_env/ChessWorld-v0")

register(
    id="chess_env/BishopWorld-v0",
    entry_point="chess_env.envs:BishopWorldEnv",
)

register(
    id="chess_env/CamelWorld-v0",
    entry_point="chess_env.envs:CamelWorldEnv",
)

# register( #FIXME: this environment's movement needs to be fixed before it can be used
#     id="chess_env/DragonHorseWorld-v0",
#     entry_point="chess_env.envs:DragonHorseWorldEnv",
# )

# register( #FIXME: this environment's movement needs to be fixed before it can be used
#     id="chess_env/DragonKingWorld-v0",
#     entry_point="chess_env.envs:DragonKingWorldEnv",
# )

register(
    id="chess_env/GoldGeneralWorld-v0",
    entry_point="chess_env.envs:GoldGeneralWorldEnv",
)

register(
    id="chess_env/KingWorld-v0",
    entry_point="chess_env.envs:KingWorldEnv",
)

register(
    id="chess_env/KnightWorld-v0",
    entry_point="chess_env.envs:KnightWorldEnv",
)

register(
    id="chess_env/SilverGeneralWorld-v0",
    entry_point="chess_env.envs:SilverGeneralWorldEnv",
)

register(
    id="chess_env/ZebraWorld-v0",
    entry_point="chess_env.envs:ZebraWorldEnv",
)