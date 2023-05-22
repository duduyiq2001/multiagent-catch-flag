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


class Buffer:
    def __init__(self,gamma):
        self.states = [] #three d arrays of integers
        self.actions = []  #integer
        self.advantages = [] #float
        self.rewards = []  #float
        self.cum_returns = [] #float
        self.state_values = [] #float
        self.logprobability = [] #float
        self.gamma = gamma
        self.pointer, self.trajectory_start = 0, 0
    def add(self, obs,action, reward, logprob):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(rewards)
        self.logprob.append(logprob)
        self.pointer += 1
    def normalize_adv(self):
        advnp = np.array(self.advantages)
        mean = np.mean(advnp)
        std = np.std(advnp)
        normalized = np.array([(adv-mean)/std for adv in advnp])
        return normalized
    def finish_eps(self):
        rewards = self.rewards[self.trajectory_start,self.pointer]
        values = self.state_values[self.trajectory_start,self.pointer]
        temp_dif = rewards[0:-1] + values[1:]*self.gamma - values[0:-1]
        ## calculate 
        cum_rewards = [0 for i in range(len(rewards))]
        cum_advantages = [0 for i in range(len(temp_dif))]
        for j in reversed(range(len(rewards))):
            if j+1 < reward_len:
                cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*self.gamma)
            else:
                cum_rewards[j] = rewards[j]
        for j in reversed(range(len(rewards))):
            if j+1 < reward_len:
                cum_advantages[j] = temp_dif[j] + (cum_advantages[j+1]*self.gamma)
            else:
                cum_advantages[j] = temp_dif[j]
        self.advantages += cum_advantages
        self.cum_returns += cum_rewards
    def get_everything(self):
        normalized_adv = self.normalizeadv()
        return (normalized_adv,np.array(self.states),np.array(self.actions), np.array(self.cum_returns),np.array(self.logprobability))
        


         
       


#actor_func = ActorNet().to(device)
#value_func = ValueNet().to(device)

def main():

    #define hyperparameters here
    batch_size = 20
    gamma = 0.99

    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    
    actnet1 = ActorNet(5).to(device)
    critnet1 = ValueNet(5).to(device)
    actnet2 = ActorNet(5).to(device)
    critnet2 = ValueNet(5).to(device)


    for batchnum in range(batch_size):
        buffer1 = Buffer(gamma)
        buffer2 = Buffer(gamma)
        nb_agents = len(env.agents)
        obs = env.reset()
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
            obs, rewards, done, _ = env.step(ac)
        

      
       

            if done:
                break

if __name__ == "__main__":
    main()