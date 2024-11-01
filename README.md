# Multi-Agent Capture the Flag: Advancing Multi-Agent Reinforcement Learning in Competitive Environments

<div align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-red"> </a>
    <img src="https://img.shields.io/badge/AI%20Powered-🤖-purple">
</div>

## Project Overview
This project presents a **competitive multi-agent reinforcement learning** approach applied to a **2 vs. 1 Capture the Flag game** environment. Utilizing **Q-learning** and **PPO** algorithms, our environment trains two collaborative agents to capture their target flag before a single adversary agent captures theirs. This research contributes to multi-agent reinforcement learning (MARL) by developing new strategies and optimized models within a custom grid-based environment.

## Objectives
The primary goal of this project is to refine MARL techniques in a mixed competitive environment, where:
- Two player agents collaborate to capture their target flag.
- One adversary agent competes by attempting to capture its own flag first.
- Agents learn optimal strategies for coordination, adversarial play, and efficient navigation using Q-learning and PPO algorithms.

## Game Environment Setup
Our custom environment is a grid-based **Capture the Flag** game, with unique features that enhance training complexity:
- **Grid Size**: 5x5 grid representing the playing field.
- **Team Flags**: Two blue flags for the 2-player team, one green flag for the adversary.
- **Action Space**: Actions include moving forward, turning left or right, and picking up the flag.
- **Blocking Mechanism**: Prevents two agents from occupying the same grid cell, adding strategic depth.
- **Reward Structure**: Sparse rewards are given only when a flag is captured, requiring agents to strategically maximize positive rewards through cumulative future gains.

## Key Technical Components

### Algorithms Used
- **Q-learning**: Optimized through hyperparameter tuning and environment simplification, enabling effective adversarial training.
- **Proximal Policy Optimization (PPO)**: Implemented to enhance the 2-agent team’s coordination and capture abilities against a randomly acting adversary.

### State Space Encoding
Our state representations evolved through four versions to improve agent performance:
1. **Partial Observation**: A 7x7 grid with 4-channel state encoding.
2. **Full Observation**: A 10x10 grid, including the entire environment.
3. **Positional Array**: Position and orientation data of each agent, reduced to a 12-element array.
4. **Enhanced Positional Array**: Added flag status indicators to track if a team’s flag is captured, increasing state detail and agent decision-making effectiveness.

### Environment Modifications
We adapted and optimized our environment, enabling a robust, strategic training ground:
- Enhanced grid blocking and movement constraints.
- Custom action and observation space encoding.
- Reward structure adjustments to reinforce capture success and reduce step count penalties.

## Experiments and Results
- **Training Runs**: We experimented with 40 training runs using stable_baselines3 DQN and PPO implementations, optimizing parameter settings for best performance.
- **Reward Stabilization**: With Q-learning, adversary agents achieved consistent capture of the flag, displaying strategic movement in simulated test runs.
- **Performance Metrics**: Rewards were tracked by cumulative score improvement per episode, and episode lengths reduced significantly, reflecting agents’ success in efficiently navigating to their targets.

## Technical Stack
- **Frameworks**: Pytorch, stable_baselines3, OpenAI Gym.
- **Visualization**: Matplotlib for reward graphs, TensorBoard for tracking training progress.
- **Environment**: Custom Capture the Flag extension built on OpenAI Gym’s multi-agent environments.
- **Recording**: Pyvirtualdisplay for video rendering and assessment.

## Codebase and File Structure
| Script               | Functionality                                                                                  |
|----------------------|------------------------------------------------------------------------------------------------|
| `best_response.py`   | Trains the adversary using a self-implemented DQN algorithm.                                   |
| `1pPPO.py`           | Implements PPO for adversary training with partial observation.                                |
| `PPO.py`             | Trains the 2-agent team with PPO against a randomly moving adversary.                          |
| `bsdqn.py`           | Uses stable_baselines3 DQN to train adversary individually.                                    |
| `qlearning.py`       | Self-implemented Q-learning algorithm for adversary training.                                  |
| `test2player.py`     | Tests and renders the 2-player team’s performance.                                             |
| `multigrid`          | Custom environment based on gym-multigrid for Capture the Flag setup.                          |

## Libraries and Dependencies
- **Pytorch**: Used for building DQN, PPO, and Q-learning models.
- **stable_baselines3**: Provided tested DQN and PPO algorithms.
- **Matplotlib & TensorBoard**: For visualization of training progress.
- **OpenAI Gym**: Multi-agent environment basis.
- **Additional Libraries**: `pyvirtualdisplay`, `numpy`, `gym==0.10.2`.

## Conclusion
Our work in advancing multi-agent reinforcement learning has demonstrated that traditional algorithms like Q-learning remain effective under certain conditions. By optimizing our environment and training techniques, we achieved meaningful progress in the Capture the Flag setup. Future work will explore more robust policy gradient methods to enhance the stochastic exploration capabilities of each agent, with the aim of fostering dynamic adaptability in competitive and cooperative MARL scenarios.
