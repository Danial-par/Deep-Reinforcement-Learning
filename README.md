# Deep-Reinforcement-Learning
Collection of 14 Reinforcement Learning course homeworks covering foundations, value-based, policy-based, actor-critic, model-based, bandits, exploration, imitation &amp; inverse RL, offline, multi-agent, hierarchical, and meta-RL. Each with notebooks, questions, and solutions.


### 1. [Introduction to RL](01_Introduction_to_RL/)

This homework introduces the fundamentals of reinforcement learning through practical exercises. It involves solving predefined environments (e.g., CartPole, FrozenLake, Taxi) using PPO and DQN from Stable Baselines3, and designing a custom 4×4 GridWorld environment in Gymnasium. A short Pygame tutorial is also included to demonstrate environment creation from scratch.

### 2. [Value-Based Methods](02_Value_Based_Methods/)
This homework focuses on value-based reinforcement learning. It implements and analyzes N-Step SARSA and N-Step Q-Learning on the Cliff Walking environment, exploring the effects of ε-greedy exploration and multi-step updates. It also compares DQN and DDQN on CartPole, highlighting the reduction of overestimation bias and improved training stability.

### 3. [Policy-Based Methods](03_Policy_Based_Methods/)
This homework explores policy-based reinforcement learning, focusing on the REINFORCE algorithm and its variants. It compares REINFORCE with Genetic Algorithms (GA) for policy search, studies the effect of baselines on training stability in CartPole, extends REINFORCE to continuous action spaces with MountainCarContinuous, and contrasts Policy Gradients with DQN on Frozen Lake.

### 4. [Advanced Methods](04_Advanced_Methods/)
This homework explores advanced actor-critic algorithms in deep reinforcement learning, including Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient (DDPG), and Soft Actor-Critic (SAC). Each method is implemented and analyzed across continuous-control environments such as HalfCheetah, focusing on sample efficiency, stability, and exploration strategies. The final part compares all three algorithms, highlighting SAC’s superior balance between stability and performance.

### 5. [Model-Based RL](05_Model_Based_RL/)
This homework focuses on model-based reinforcement learning techniques, combining planning, learning, and control. It begins with implementing a MuZero-inspired agent using Monte Carlo Tree Search (MCTS) for decision-making, followed by the Dyna-Q algorithm and Prioritized Sweeping on the FrozenLake environment. The final section explores Model Predictive Control (MPC) to solve the Pendulum environment using differentiable optimization in PyTorch. The exercises emphasize planning efficiency, reward shaping, and the trade-off between learned dynamics and control precision.

### 6. [Multi-Armed Bandits](06_Multi_Armed_Bandits/)
This homework explores multi-armed bandit (MAB) algorithms, focusing on the trade-off between exploration and exploitation. It implements and analyzes several bandit agents — including Oracle, Random, Explore-First, UCB, Epsilon-Greedy, and LinUCB — across stationary and contextual settings. The project culminates in a deep conceptual analysis of regret, non-stationarity, hyperparameter sensitivity, and hybrid strategies, connecting theoretical regret guarantees with practical performance in finite-horizon environments.