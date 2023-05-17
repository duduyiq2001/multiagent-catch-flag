import gym
from gym_multigrid.envs.collect_game import CollectGameEnv

class MyCustomCollectGameEnv(CollectGameEnv):
    def __init__(self):
        super().__init__(
            size=10,
            width= 11,
            height = 11,
            num_balls=[2, 3],
            agents_index=[1, 2],
            balls_index=[3, 4],
            balls_reward=[1, 2],
            zero_sum=True,
            view_size=7
        )

# Register the custom environment with OpenAI Gym
gym.envs.register(
    id="MyCustomCollectGame-v0",
    entry_point="my_custom_env:MyCustomCollectGameEnv",
)