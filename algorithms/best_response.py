import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

from gym.envs.registration import register

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#sys.path.append(r"C:\Users\Administrator\Desktop\multiagent-catch-flag")
sys.path.append(r"../")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pruneobs(obs):
    newarray = np.zeros(shape=(7, 7, 4), dtype=np.uint8)
    for a in range(7):
        for b in range(7):

            newarray[a][b] = np.concatenate((obs[a][b][0:2],obs[a][b][4:6]), axis =0)
    return newarray

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, num_actions):

        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(32*3*3, 64),
            nn.Linear(64, num_actions),
        )
        self.conv.to(device)



    # def _get_conv_out(self, shape):
    #     o = self.conv(torch.zeros(1, *shape))
    #     return int(np.prod(o.size()))

    def forward(self, x):

        x = self.conv(x)
        return x
# Define the DQN Agent
class DQNAgent:
    def __init__(self, num_actions, lr, gamma, epsilon_max, epsilon_min, epsilon_decay, buffer_capacity, batch_size):
        self.num_actions = num_actions
        self.q_network = QNetwork( num_actions)
        self.target_network = QNetwork( num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer = []
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.epsilon = epsilon_max

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state = torch.cuda.FloatTensor(np.array(state)).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()
        
    def update_epsilon(self, episode):
        epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (episode / self.epsilon_decay))
        self.epsilon = epsilon
        
    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert numpy arrays to PyTorch tensors
        states = torch.cuda.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.cuda.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(np.array(dones)).to(device)
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1).to(device)
        next_q_values = self.target_network(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        expected_q_values = rewards + self.gamma * max_next_q_values * (~dones)   

        loss = self.loss(q_values, expected_q_values)

        # q_network_params_before = [param.data.clone() for param in self.q_network.parameters()]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Check if the parameter tensors have changed
        # q_network_params_after = [param.data for param in self.q_network.parameters()]
        # params_changed = any(not torch.allclose(param_before, param_after) for param_before, param_after in zip(q_network_params_before, q_network_params_after))

        # if params_changed:
        #     print("DQN parameters have changed.")
        # else:
        #     print("DQN parameters have not changed.")
        


# Define the main training loop
def main():
    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')
    # state_shape = env.observation_space.shape
    # state_shape = [1,7,7,4]
    num_actions = env.action_space.n
    lr = 0.1
    gamma = 0.99
    epsilon_max = 0.9
    epsilon_min = 0.01
    epsilon_decay = 500
    buffer_capacity = 10000
    batch_size = 32
    adversary = DQNAgent(num_actions, lr, gamma, epsilon_max, epsilon_min, epsilon_decay, buffer_capacity, batch_size)
    num_episodes = 1000
    eps_returns = []
    eps_step =[]
    for episode in range(num_episodes):  
        states = env.reset()
        #print(env.remaining_ball)
        done = False
        total_reward = 0 
        steps = 0      
        while not done:
            # env.render(mode='human', highlight=True)
            newobs = [pruneobs(agent) for agent in states] ##use newobs
            newobs[0] = np.transpose(newobs[0], (2, 0, 1))           
            actions = []
            steps += 1
            # Select actions for the adversary agent only
            actions.append(adversary.select_action(newobs[0]))

            for _ in range(1, len(newobs)):
                actions.append(env.action_space.sample())

            states, rewards, done, _ = env.step(actions)
            newobs1 = [pruneobs(agent) for agent in states] ##use newobs
            newobs1[0] = np.transpose(newobs1[0], (2, 0, 1))
            # Buffer experiences and update only for the adversary agent
            total_reward += rewards[0] 
            adversary.buffer.append((newobs[0], actions[0], rewards[0], newobs1[0], done))       
        adversary.update()      
        # Update the target network every few episodes
        if episode % 5 == 0:
            adversary.target_network.load_state_dict(adversary.q_network.state_dict())
        if episode % 100 == 1:
            torch.save(adversary.target_network.state_dict(), f'adversary{episode}.pt')

        time.sleep(5)
        eps_returns.append(total_reward)
        eps_step.append(steps)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")
        if episode%20 == 10:
            
            print(f'current return {sum(eps_returns[-10:])/10}')
            print(f'current steps {sum(eps_step[-10:])/10}')


        # Update DQN here
        adversary.update_epsilon(episode)
    env.close()
# Start training
main()