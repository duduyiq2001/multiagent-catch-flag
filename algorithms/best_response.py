import gym
import time
from gym.envs.registration import register
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#sys.path.append(r"C:\Users\Administrator\Desktop\multiagent-catch-flag")
sys.path.append(r"../")




#parser = argparse.ArgumentParser(description=None)
#parser.add_argument('-e', '--env', default='c', type=str)

#args = parser.parse_args()
def pruneobs(obs):
    newarray = np.zeros(shape=(7, 7, 4), dtype=np.uint8)
    for a in range(7):
        for b in range(7):
            #print(len(obs[a][b]))
            newarray[a][b] = np.concatenate((obs[a][b][0:2],obs[a][b][4:6]), axis =0)
    return newarray
class QNetwork(nn.Module):
    def __init__(self,action_size):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.Flatten(),
        nn.Linear(32*3*3, 64),
        nn.Linear(64, action_size),

        )

    def forward(self, x):
        return self.network(x)

def main():

    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    obs = env.reset()
    print(obs[0])
    qnet = QNetwork(5)

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        #time.sleep(0.1)

        newobs = [pruneobs(agent) for agent in obs] ##use newobs
        print(newobs[0])
        action_list = qnet.forward(newobs[0])
        action = action_list.argmax().item()
        ac_of_team = [env.action_space.sample() for _ in range(nb_agents-1)]
        ac = [action] + ac_of_team
        obs, _, done, _ = env.step(ac)
      
       

        if done:
            break

if __name__ == "__main__":
    main()