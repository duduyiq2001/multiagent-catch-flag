
import gym
import time
from gym.envs.registration import register
import argparse
import sys
#sys.path.append(r"C:\Users\Administrator\Desktop\multiagent-catch-flag")
sys.path.append(r"../")


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='c', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    else:
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _ = env.step(ac)
        print(obs[0])
        #print(obs)

        if done:
            break

if __name__ == "__main__":
    main()