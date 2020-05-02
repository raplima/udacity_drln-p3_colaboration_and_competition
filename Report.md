## Learning Algorithm

The agent is based on [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) 
with simple (random) replay memory. Two agents (one for each racket) share a replay buffer. As noticed in the [paper](https://arxiv.org/abs/1706.02275) multi-agent training is hard when using traditional algorithms: Q-learning is challenged by an inherent non-stationarity of the environment, while policy gradient suffers from a variance that increases as the number of agents grows. [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) 
presents an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multiagent coordination
Here, two agents are used to solve the environment proposed. These actions perform similar actions according to similar observations.
Following steps in the [paper](https://arxiv.org/abs/1706.02275), I adapt the actor-critic method to consider action policies of other agents by adopting a centralized training with decentralized execution. During training, each agent has its own critic network, that takes as input the observations and actions of any agent present in the environment (centralized training). Each agent has its own actor network that takes actions considering only its observation of the environment.  
I experimented adding [Ornstein–Uhlenbeck noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) to the agents' actions, but I did not notice a significant improvement with the addition of noise, so I removed the noise option. 
The agent minimizes the MSE loss computed between expected and actual rewards using [Adam](https://arxiv.org/abs/1412.6980). 
The weights of the agent's networks are softly updated every four training steps. 
Other hyperparameters are given below (chosen by experimentation):

```python
LR_ACTOR = 1e-3                 # Learning rate of actor model
LR_CRITIC = 1e-3                # Learning rate of critic model
TAU = 1e-2                      # Soft Update Parameter
GAMMA = 0.99                    # Discount Factor
BUFFER_SIZE = int(1e6)          # Replay buffer size
BATCH_SIZE = 512                # Batch size for training models
RANDOM_SEED = 0                 # Seeding
WEIGHT_DECAY = 0.               # Weight decay for Critic Model
UPDATE_EVERY = 4                # model update frequency
```

The agent's Actor network is composed of two hidden layers with batch normalization and leaky ReLU activation. 
Both layers contain 64 neurons. The output layer uses an tanh activation
The agent's Critic network is composed of three hidden layers with batch normalization and leaky ReLU activation. 
The three layers contain 64 neurons. The input for the first layer is the state for all the actors. 
The actions are concatenated in the second layer. Other Udacity students applied similar techniques to solve the environment, for example
[here](https://github.com/Kushagra14/Collabration_and_Competition), 
[here](https://github.com/AlessandroRestagno/Collaboration-and-competition-DRLND-P3-Udacity), 
and [here](https://github.com/silviomori/udacity-deep-reinforcement-learning-p3-collab-compet/). 
The choice for network architecture is based on experimentation aiming to solve the environment as well as an attempt to mantain the architecture simple. 

## Plot of Rewards
![alt text](./training.pdf "Rewards per episode - the agent receives an average reward (over 100 episodes) of at least +0.5. ")  
The environment was solved in 1622 episodes. Note the _*version 1 (Tennis)*_ is solved. 

## Ideas for Future Work

This project uses [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) 
and it is able to solve the environment in less than 2000 episodes. However, the performance seems instable. 
In some experiments with similar hyperparameters the agents are unable to solve the environment in 3000 episodes. 
Although I performed some hyperparameter tuning, it is likely a better choice of hyperparameter could lead to more stable agents. 
The performance might be improved with other methods, such as:
* [Trust Region Policy Optimization)](https://arxiv.org/abs/1502.05477)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617)  
Check [this paper](https://arxiv.org/abs/1604.06778) for a discussion of Deep Reinforcement Learning for Continuous Control.
A [prioritised experience buffer](https://github.com/Damcy/prioritized-experience-replay) may be helpful for improving the performance. 
