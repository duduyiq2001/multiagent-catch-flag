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
    for i in range(2):
        _ = env.reset()

        nb_agents = len(env.agents)
        total0 = 0
        total1 = 0
        total2 = 0

        while True:
            env.render(mode='human', highlight=True)

            #ac = [env.action_space.sample() for i in range(3)]
            ac = input('actions?\n')
            ac = ac.split(" ")
            ac = [eval(num) for num in ac]

            print(ac)

            obs, rewards, done, _ = env.step(ac)
            print(obs)

            print(f'reward {rewards}')
            total0 += rewards[0]
            total1 += rewards[1]
            total2 += rewards[2]
            print(f'total{total0} {total1} {total2}')
        
            #print(obs)

            if done:
                break

if __name__ == "__main__":
    main()