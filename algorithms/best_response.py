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


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_shape, num_actions):

        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[1], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = self._get_conv_out(state_shape[1:])
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        q_values = self.fc(conv_out)
        return q_values

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_shape, num_actions, lr, gamma, epsilon_max, epsilon_min, epsilon_decay, buffer_capacity, batch_size):
        self.num_actions = num_actions
        self.q_network = QNetwork(state_shape, num_actions)
        self.target_network = QNetwork(state_shape, num_actions)
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
            state = torch.from_numpy(state).unsqueeze(0).float()
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

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_network(next_states).detach()
        max_next_q_values = next_q_values.max(1)[0]
        expected_q_values = rewards + self.gamma * max_next_q_values * ~dones

        loss = self.loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Define the main training loop
def main():

    register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
    env = gym.make('multigrid-collect-v0')

    # state_shape = env.observation_space.shape
    state_shape = [1,7,7,6]

    num_actions = env.action_space.n
    lr = 0.1
    gamma = 0.99
    epsilon_max = 0.9
    epsilon_min = 0.01
    epsilon_decay = 500
    buffer_capacity = 10000
    batch_size = 32

    adversary = DQNAgent(state_shape, num_actions, lr, gamma, epsilon_max, epsilon_min, epsilon_decay, buffer_capacity, batch_size)

    num_episodes = 1000
    for episode in range(num_episodes):
    
        states = env.reset()
        print(env.remaining_ball)

        done = False
        total_reward = 0

        while not done:
            env.render(mode='human', highlight=True)

            actions = []
            for agent_id, state in enumerate(states):
                if agent_id == 0:
                    actions.append(adversary.select_action(state))
                else:
                    actions.append(env.action_space.sample())

            next_states, rewards, done, _ = env.step(actions)

            # Buffer experiences and update only for the trained agent
            if env.remaining_ball <= 0:
                adversary.buffer.append((states[0], actions[0], rewards[0], next_states[0], done))
                adversary.update()
                done = True

            states = next_states
            total_reward += rewards[0] 

        # Update the target network every few episodes
        if episode % 5 == 0:

            before_update_params = [param.clone() for param in adversary.q_network.parameters()]

            adversary.target_network.load_state_dict(adversary.q_network.state_dict())

            after_update_params = [param.clone() for param in adversary.q_network.parameters()]

            # Compare the parameters
            for param1, param2 in zip(before_update_params, after_update_params):
                if not torch.equal(param1, param2):
                    print("DQN parameters have changed.")
            time.sleep(5)


        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

        # Update DQN here
        adversary.update_epsilon(episode)

        

    env.close()

# Start training
main()