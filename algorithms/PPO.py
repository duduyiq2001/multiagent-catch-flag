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
    batch_size = 5 ## 20 times 300 = 6000 episodes
    gamma = 0.99
    epoch = 300
    train_iter = 1
    clip_ratio = 0.2


    ###
    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    
    actnet1 = ActorNet(5).to(device)
    critnet1 = ValueNet().to(device)
    actnet2 = ActorNet(5).to(device)
    critnet2 = ValueNet().to(device)
    ### set up buffers
    
    nb_agents = len(env.agents)
    eps_return1 = []
    eps_return2 = []
    eps_steps = []
    opt1act = torch.optim.AdamW(actnet1.parameters(), lr=0.1)
    opt2act = torch.optim.AdamW(actnet2.parameters(), lr=0.1)
    opt1val = torch.optim.AdamW(critnet1.parameters(), lr=0.1)
    opt2val = torch.optim.AdamW(critnet2.parameters(), lr=0.1)
    for ep in range(epoch):
        buffer1 = Buffer(gamma)
        buffer2 = Buffer(gamma)
        for batchstep in range(batch_size):
            torch.autograd.set_detect_anomaly(True)
            obs = env.reset()
            sum_reward1 = 0
            sum_reward2 = 0
            steps = 0
            while True:          
                #env.render(mode='human', highlight=True)
                #time.sleep(0.1)
                newobs = [pruneobs(agent) for agent in obs] ##use newobs
                newobs[1] = np.transpose(newobs[1], (2, 0, 1))
                newobs[2] = np.transpose(newobs[2], (2, 0, 1))
                value1 = critnet1.forward( torch.from_numpy(newobs[1]).float().unsqueeze(0).to(device))
                value2 = critnet2.forward(torch.from_numpy(newobs[2]).float().unsqueeze(0).to(device))
                #print(f'value {value1} {value1.type}')
                # Convert to a PyTorch tensor
                ac_of_adv = env.action_space.sample()
                ac_of_p1,logprob1 = actnet1.sample( torch.from_numpy(newobs[1]).float().unsqueeze(0).to(device))

                #print(f'ac_of_p1{ac_of_p1} {logprob1}')
                #return (a.tolist()[0],logprobs[a.tolist()[0]].item())
                ac_of_p2,logprob2 = actnet2.sample( torch.from_numpy(newobs[2]).float().unsqueeze(0).to(device))
                ac = [ac_of_adv, ac_of_p1,ac_of_p2]
                obs, rewards, done, _ = env.step(ac)
                sum_reward1 += rewards[1]
                sum_reward2 += rewards[2]
                steps += 1
                buffer1.add(newobs[1],ac_of_p1,rewards[1],value1.item(),logprob1) 
                buffer2.add(newobs[2],ac_of_p2,rewards[2],value2.item(),logprob2) 
                #print(buffer1.state_values)      
                if done:
                    newobs = [pruneobs(agent) for agent in obs] ##use newobs
                    newobs[1] = np.transpose(newobs[1], (2, 0, 1))
                    newobs[2] = np.transpose(newobs[2], (2, 0, 1))
                    value1 = critnet1.forward( torch.from_numpy(newobs[1]).float().unsqueeze(0).to(device))
                    value2 = critnet2.forward( torch.from_numpy(newobs[2]).float().unsqueeze(0).to(device))
                    buffer1.finish_eps(value1.item())
                    buffer2.finish_eps(value2.item())
                    eps_return1.append(sum_reward1)
                    eps_return2.append(sum_reward2)
                    eps_steps.append(steps)
                    print(f'episode{batchstep}agent 1 return {sum_reward1} agent 2 return {sum_reward2} in {steps}')
                    break
        print(f'batch return')
        # get all batch info 
        cum_adv1, states1, actions1,cum_returns1,logprobs1,values1 = buffer1.get_everything()
        cum_adv2, states2, actions2,cum_returns2,logprobs2,values2 = buffer2.get_everything()

        print(len(cum_adv1))
        print(len(states1))
        print(len(actions1))
        print(len(cum_returns1))
        print(len(logprobs1))
        #convert all of them to 
        logprobs_1 = actnet1.get_logprob(states1, actions1)
        logprobs_2 = actnet2.get_logprob(states2, actions2)
        #print(logprobs_2)
        # train with 
        torch.autograd.set_detect_anomaly(True)
        for i in range(train_iter):
            kl1 = actnet1.train_policy(
            states1,actions1 , logprobs_1 , cum_adv1,opt1act,clip_ratio
            )
            torch.autograd.set_detect_anomaly(True)
            kl2 = actnet2.train_policy(
            states2,actions2 , logprobs_2, cum_adv2,opt2act,clip_ratio
            )
            print(f'kl divergence at iter{i} is {kl1} and {kl2}')
        #print("cum_rewards.requires_grad: ", cum_returns1.requires_grad)
        #print("values.requires_grad: ", values1.requires_grad)
        for a in range(train_iter):
            critnet1.train(cum_returns1,states1,opt1val)
            critnet2.train(cum_returns2,states2,opt2val)
        if ep%10 == 1:
            #save model
            torch.save(actnet1.state_dict(),f'epoch_{ep}_actnet1.pt')
            torch.save(actnet2.state_dict(),f'epoch_{ep}_actnet2.pt')
            torch.save(critnet1.state_dict(),f'epoch_{ep}_critnet1.pt')
            torch.save(critnet2.state_dict(),f'epoch_{ep}_critnet2.pt')
        if ep%10 == 0:
            record_video(env,actnet1,actnet2,f'epoch{ep}.mp4')
        

        
if __name__ == "__main__":
    main()