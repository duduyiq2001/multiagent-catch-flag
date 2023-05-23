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
        #print(x.shape)
        return self.network(x)
    def sample(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits).squeeze(0)
        logprobs = torch.log(probs)
        #print(f'probs{probs}')
        #print(logprobs)
        a = torch.multinomial(probs, num_samples=1)
        #print(a)

        return (a.tolist()[0],logprobs[a.tolist()[0]])
  






class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.Flatten(),
        nn.Linear(32*3*3, 64),
        nn.Linear(64, 1),)
    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0).float()
        x = x.clone().detach().to(device) 
        return self.network(x)[0][0].item()
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
        self.logprob = [] #float
        self.gamma = gamma
        self.pointer, self.trajectory_start = 0, 0
    def add(self, obs,action, reward,statevalue, logprob):
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_values.append(statevalue)
        self.logprob.append(logprob)
        self.pointer += 1
    def normalize_adv(self):
        advnp = np.array(self.advantages)
        mean = np.mean(advnp)
        std = np.std(advnp)
        normalized = np.array([(adv-mean)/std for adv in advnp])
        return normalized
    def finish_eps(self,lastvalue):
        rewards = self.rewards[self.trajectory_start:self.pointer] +[lastvalue]
        values = self.state_values[self.trajectory_start:self.pointer] + [lastvalue]
        #print(f'values {values}')
        rewards1 = np.array(rewards)
        values1 = np.array(values)
        temp_dif = rewards1[0:-1] + values1[1:]*self.gamma - values1[0:-1]
        #print(temp_dif)
        ## calculate 
        cum_rewards = [0 for i in range(len(rewards)-1)]
        cum_advantages = [0 for i in range(len(temp_dif))]
        for j in reversed(range(len(rewards)-1)):
            if j+1 < len(rewards)-1:
                cum_rewards[j] = rewards[j-1] + (cum_rewards[j+1]*self.gamma)
            else:
                cum_rewards[j] = rewards[j-1]
        for j in reversed(range(len(rewards)-1)):
            if j+1 < len(rewards)-1:
                cum_advantages[j] = temp_dif[j] + (cum_advantages[j+1]*self.gamma)
            else:
                cum_advantages[j] = temp_dif[j]
        self.advantages += cum_advantages
        self.cum_returns += cum_rewards
    def get_everything(self):
        normalized_adv = self.normalizeadv()
        return (normalized_adv,np.array(self.states),np.array(self.actions), np.array(self.cum_returns),np.array(self.logprob))
        


         
       


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
    critnet1 = ValueNet().to(device)
    actnet2 = ActorNet(5).to(device)
    critnet2 = ValueNet().to(device)


    for batchstep in range(batch_size):
        buffer1 = Buffer(gamma)
        buffer2 = Buffer(gamma)
        nb_agents = len(env.agents)
        obs = env.reset()
        eps_return1 = []
        eps_return2 = []
        eps_steps = []
        sum_reward1 = 0
        sum_reward2 = 0
        steps = 0
        while True:
            
            #env.render(mode='human', highlight=True)
            #time.sleep(0.1)

            newobs = [pruneobs(agent) for agent in obs] ##use newobs
            newobs[1] = np.transpose(newobs[1], (2, 0, 1))
            newobs[2] = np.transpose(newobs[2], (2, 0, 1))
            value1 = critnet1.forward(newobs[1])
            value2 = critnet2.forward(newobs[2])
            #print(f'value {value1} {value1.type}')


            # Convert to a PyTorch tensor
            ac_of_adv = env.action_space.sample()
            ac_of_p1,logprob1 = actnet1.sample(newobs[1])

            #print(f'ac_of_p1{ac_of_p1} {logprob1}')
            ac_of_p2,logprob2 = actnet2.sample(newobs[2])
            ac = [ac_of_adv, ac_of_p1,ac_of_p2]
            obs, rewards, done, _ = env.step(ac)
            sum_reward1 += rewards[1]
            sum_reward2 += rewards[2]
            steps += 1


            buffer1.add(newobs[1],ac_of_p1,rewards[1],value1,logprob1) 
            buffer2.add(newobs[2],ac_of_p2,rewards[2],value2,logprob2) 
            #print(buffer1.state_values)      
            if done:
                newobs = [pruneobs(agent) for agent in obs] ##use newobs
                newobs[1] = np.transpose(newobs[1], (2, 0, 1))
                newobs[2] = np.transpose(newobs[2], (2, 0, 1))
                value1 = critnet1.forward(newobs[1])
                value2 = critnet2.forward(newobs[2])
                buffer1.finish_eps(value1)
                buffer2.finish_eps(value2)
                eps_return1.append(sum_reward1)
                eps_return2.append(sum_reward2)
                eps_steps.append(steps)
                print(f'episode{batchstep}agent 1 return {sum_reward1} agent 2 return {sum_reward2} in {steps}')
                break
    print(f'batch return')

      
       

           

if __name__ == "__main__":
    main()