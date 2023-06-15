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
print(device)
time.sleep(5)
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
        #x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        #x = x.clone().detach().to(device) #dimension might not work
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
    def get_logprob(self,states,actions):
        logits = self.network(states)
        probs = F.softmax(logits)
        #print(f'probs{probs}')
        logprobs = torch.log(probs)
        #print(logprobs.type)
        associated_probs = logprobs[torch.arange(logprobs.size(0)), actions]
        
        #print(f'logprobs {associated_probs}')
        return associated_probs
    def train_policy(self,obs, actions, logprobabilities, advantages,optimizer,clip_ratio):
        #empty the optimzier
        optimizer.zero_grad()
        # calcualtion for policy los
        post_prob = self.get_logprob(obs, actions)
        ratio = torch.exp(
            post_prob
            - logprobabilities
        )
        min_advantage = torch.where(
            advantages > 0,
            (1 + clip_ratio) * advantages,
            (1 - clip_ratio) * advantages,
        )

        policy_loss = -torch.mean(
            torch.min(ratio * advantages, min_advantage)
        )
        policy_loss.backward()
        optimizer.step() 

        kl = torch.mean(
            logprobabilities
            - self.get_logprob(obs, actions)
        )
        return kl.sum()
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
        #x = torch.from_numpy(x).unsqueeze(0).float().to(device)
        #x = x.clone().detach().to(device) 
        return self.network(x)[0][0]
    def train(self,cum_rewards,states,optimizer):
        values = self.network(states)
        #print(values)
        optimizer.zero_grad()
        vf_loss = torch.mean((cum_rewards - values) ** 2)
        vf_loss.sum().backward()
        optimizer.step()


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
        adv_tensor = torch.tensor(self.advantages, dtype=torch.float32).to(device)
        mean = torch.mean(adv_tensor)
        std = torch.std(adv_tensor)
        normalized = (adv_tensor - mean) / std
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
        self.trajectory_start = self.pointer
    def get_everything(self):
        normalized_adv = self.normalize_adv()
        states_tensor = torch.tensor(self.states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(self.actions, dtype=torch.int64).to(device)
        cum_returns_tensor = torch.tensor(self.cum_returns, dtype=torch.float32).to(device)
        logprob_tensor = torch.tensor(self.logprob, dtype=torch.float32).to(device)
        state_values_tensor = torch.tensor(self.state_values, dtype=torch.float32).to(device)
        return (normalized_adv,states_tensor,actions_tensor, cum_returns_tensor,logprob_tensor,state_values_tensor)
        # adv, states, actions,cum_returns,logrobs
def record_video(env,actnet1, actnet2,out_directory, fps=30):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we
    """
    images = []
    done = False
    state = env.reset()
    img = env.render(mode='rgb_array')
    images.append(img)
    counter = 0
    while not done:
        # Take the action (index) that have the maximum expected future reward g
        #action, _ = policy.act(state)
        adv = env.action_space.sample() 
        print(counter)
        counter +=1
        ac_of_p1,logprob1 = actnet1.sample( torch.from_numpy(newobs[1]).float().unsqueeze(0).to(device))

                #print(f'ac_of_p1{ac_of_p1} {logprob1}')
                #return (a.tolist()[0],logprobs[a.tolist()[0]].item())
        ac_of_p2,logprob2 = actnet2.sample( torch.from_numpy(newobs[2]).float().unsqueeze(0).to(device))
        actions = [adv,ac_of_p1,ac_of_p2]
        state, reward, done, info = env.step(actions) # We directly put next_stat
        img = env.render(mode='rgb_array')
        images.append(img)
        imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)])
        if counter >= 200:
            break
#record_video('./replay2.mp4')
def main():
    torch.autograd.set_detect_anomaly(True)
    #define hyperparameters here
    batch_size = 3 ## 20 times 300 = 6000 episodes
    gamma = 0.99
    epoch = 30000
    train_iter = 1
    clip_ratio = 0.2


    ###
    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    
    actnet0= ActorNet(5).to(device)
    critnet0 = ValueNet().to(device)
    ### set up buffers
    
    nb_agents = len(env.agents)
    eps_return0 = []
    eps_steps = []
    opt0act = torch.optim.AdamW(actnet0.parameters(), lr=0.5)
    opt0val = torch.optim.AdamW(critnet0.parameters(), lr=0.5)
    for ep in range(epoch):
        buffer0 = Buffer(gamma)
        for batchstep in range(batch_size):
            torch.autograd.set_detect_anomaly(True)
            obs = env.reset()
            sum_reward0 = 0
            steps = 0
            while True:          
                #env.render(mode='human', highlight=True)
                #time.sleep(0.1)
                newobs = [pruneobs(agent) for agent in obs] ##use newobs
                newobs[0] = np.transpose(newobs[0], (2, 0, 1))
                value0 = critnet0.forward( torch.from_numpy(newobs[0]).float().unsqueeze(0).to(device))
                #print(f'value {value1} {value1.type}')
                # Convert to a PyTorch tensor
                ac_of_adv,logprob0 = actnet0.sample( torch.from_numpy(newobs[0]).float().unsqueeze(0).to(device))
              
                ac = [ac_of_adv]
                ac.append(env.action_space.sample())
                ac.append(env.action_space.sample())
                obs, rewards, done, _ = env.step(ac)
                sum_reward0 += rewards[0]
              
                steps += 1
                if len(buffer0.states) > 10000:
                    buffer0 = Buffer(gamma)

                buffer0.add(newobs[0],ac_of_adv,rewards[0],value0.item(),logprob0) 
                #print(buffer1.state_values)      
                if done:
                    newobs = [pruneobs(agent) for agent in obs] ##use newobs
                    newobs[0] = np.transpose(newobs[0], (2, 0, 1))
                    value0 = critnet0.forward( torch.from_numpy(newobs[0]).float().unsqueeze(0).to(device))                
                    buffer0.finish_eps(value0.item())
                    eps_return0.append(sum_reward0)
                    eps_steps.append(steps)
                    print(f'episode{batchstep}agent 0 return {sum_reward0}  in {steps}')
                    break
        print(f'batch return')
        # get all batch info 
        cum_adv0, states0, actions0,cum_returns0,logprobs0,values0 = buffer0.get_everything()
        

        #convert all of them to 
        logprobs_0 = actnet0.get_logprob(states0, actions0)
        #print(logprobs_2)
        # train with 
        torch.autograd.set_detect_anomaly(True)
        for i in range(train_iter):
            kl0 = actnet0.train_policy(
            states0,actions0 , logprobs_0 , cum_adv0,opt0act,clip_ratio
            )
            torch.autograd.set_detect_anomaly(True)
           
            print(f'kl divergence at iter{i} is {kl0}')
        #print("cum_rewards.requires_grad: ", cum_returns1.requires_grad)
        #print("values.requires_grad: ", values1.requires_grad)
        for a in range(train_iter):
            critnet0.train(cum_returns0,states0,opt0val)
        if ep%100 == 0:
            #save model
            torch.save(actnet0.state_dict(),f'epoch_{ep}_actnet0.pt')
            torch.save(critnet0.state_dict(),f'epoch_{ep}_critnet0.pt')
        if ep%20 == 10:
            print(f'current return {sum(eps_return0[-10:])/10}')
            print(f'current steps {sum(eps_steps[-10:])/10}')

        #if ep%10 == 0:
            #record_video(env,actnet0,actnet2,f'epoch{ep}.mp4')
        

if __name__ == "__main__":
  main()