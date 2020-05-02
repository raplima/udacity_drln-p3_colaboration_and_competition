## Learning Algorithm

The agent is based on [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) 
with simple (random) replay memory. 
The agent minimizes the MSE loss computed between expected and actual rewards using [Adam](https://arxiv.org/abs/1412.6980). 
The weights of the agent's networks are softly updated every four training steps. 
Other hyperparameters are given below:

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
