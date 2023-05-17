import gym
import gym_multigrid  # Import the gym_multigrid module to register the environments

# Create the environment using the environment ID (replace 'multigrid-soccer-v0' with the desired environment ID)
env = gym.make('multigrid-collect-v0')

# Initialize the environment
observation = env.reset()

# Number of episodes
num_episodes = 10

# Loop through episodes
for episode in range(num_episodes):
    done = False
    
    # Reset the environment for each episode
    observation = env.reset()

    while not done:
        # Render the environment (optional)
        env.render()

        # Choose an action (replace this with your own action selection logic)
        action = env.action_space.sample()

        # Perform the action and receive feedback (observation, reward, done, info)
        observation, reward, done, info = env.step(action)

    print(f"Episode {episode + 1} finished")

# Close the environment
env.close()
