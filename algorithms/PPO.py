import gym
import time
from gym.envs.registration import register
import argparse
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(r"../")
class ActorNet(nn.Module):
    def __init__(self,action_size):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.Flatten(),
        nn.Linear(32*3*3, 64),
        nn.Linear(64, action_size),)
    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = x.clone().detach().to(device) #dimension might not work
        print(x.shape)
        return self.network(x)
    def sample(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits)
        print(logits)
        a = torch.multinomial(probs, num_samples=1)
        print(a)

        return a.tolist()[0][0]





class ValueNet(nn.Module):
    def __init__(self,action_size):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.Flatten(),
        nn.Linear(32*3*3, 64),
        nn.Linear(64, action_size),)
    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0).float()
        x = x.clone().detach().to(device) 
        return self.network(x)
def pruneobs(obs):
    newarray = np.zeros(shape=(7, 7, 4), dtype=np.uint8)
    for a in range(7):
        for b in range(7):
            #print(len(obs[a][b]))
            newarray[a][b] = np.concatenate((obs[a][b][0:2],obs[a][b][4:6]), axis =0)
    return newarray



#actor_func = ActorNet().to(device)
#value_func = ValueNet().to(device)

def main():

    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    obs = env.reset()
    actnet1 = ActorNet(5).to(device)
    critnet1 = ValueNet(5).to(device)
    actnet2 = ActorNet(5).to(device)
    critnet2 = ValueNet(5).to(device)


    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        #time.sleep(0.1)

        newobs = [pruneobs(agent) for agent in obs] ##use newobs
        newobs[1] = np.transpose(newobs[1], (2, 0, 1))
        newobs[2] = np.transpose(newobs[2], (2, 0, 1))
        
        



# Convert to a PyTorch tensor
        ac_of_adv = env.action_space.sample()
        ac_of_p1 = actnet1.sample(newobs[1])
        print(f'ac_of_p1{ac_of_p1}')
        ac_of_p2 = actnet2.sample(newobs[2])
        ac = [ac_of_adv, ac_of_p1,ac_of_p2]
        obs, _, done, _ = env.step(ac)
      
       

        if done:
            break

if __name__ == "__main__":
    main()