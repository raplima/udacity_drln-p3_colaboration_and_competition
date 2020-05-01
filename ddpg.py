import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

LR_ACTOR = 1e-3                 # Learning rate of actor model
LR_CRITIC = 1e-3                # Learning rate of critic model
TAU = 1e-2                      # Soft Update Parameter
GAMMA = 0.99                    # Discount Factor
BUFFER_SIZE = int(1e6)          # Replay buffer size
BATCH_SIZE = 512                # Batch size for training models
RANDOM_SEED = 0                 # Seeding
WEIGHT_DECAY = 0.               # Weight decay for Critic Model
UPDATE_EVERY = 4                # model update frequency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    
    def __init__(self, state_size, action_size, index, num_agents, do_soft_update=True, do_hard_update=False):
        """Initialize a DDPGAgent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            agent_id (int): identifier for this agent
        """
        self.action_size = action_size
        self.state_size = state_size
        self.index = index
        self.num_agents = num_agents
        self.do_soft_update = do_soft_update
        self.do_hard_update = do_hard_update
        
        # set up networks
        self.actor_local = Actor(state_size, action_size, RANDOM_SEED).to(device)
        self.actor_target = Actor(state_size, action_size, RANDOM_SEED).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr = LR_ACTOR)
        
        self.critic_local = Critic(state_size, action_size, num_agents, RANDOM_SEED).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, RANDOM_SEED).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr = LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # make sure local and target networks start with same weights
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        # initialize steps counter
        self.timesteps = 0
       
    def act(self,state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        return np.clip(action,-1,1)
    
    def learn(self,experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(next_state) -> action
            critic_target(next_state, next_action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            next_actions (list): next actions computed from each agent
            actions_pred (list): prediction for actions for current states from each agent
        """
        states, actions, rewards, next_states, dones = experiences
        all_states = torch.cat(states, dim=1).to(device)
        all_next_states = torch.cat(next_states, dim=1).to(device)
        all_actions = torch.cat(actions, dim=1).to(device)
        
        # Predict next action with target actor network Next Action Prediction using Target Actor Model
        #with torch.no_grad():
        next_actions = [actions[index].clone() for index in range(self.num_agents)]
        next_actions[self.index] = self.actor_target(next_states[self.index])
        all_next_actions = torch.cat(next_actions, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            Q_target_next = self.critic_target(all_next_states, all_next_actions)
        Q_target = rewards[self.index] + GAMMA * Q_target_next *(1-dones[self.index])
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_target.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        #Action Prediction using Local Actor Model 
        actions_pred = [actions[index].clone() for index in range(self.num_agents)]
        actions_pred[self.index] = self.actor_local(states[self.index])
        all_actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        # Compute actor loss
        actor_loss = -self.critic_local(all_states, all_actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #              
        ### soft Update target every UPDATE_EVERY step
        if self.do_soft_update:
            if self.timesteps % UPDATE_EVERY == 0:
                self.soft_update(self.critic_local, self.critic_target, TAU)
                self.soft_update(self.actor_local, self.actor_target, TAU)                     
        if self.do_hard_update:
            if self.timesteps % UPDATE_EVERY == 0:
                self.hard_update(self.critic_local, self.critic_target)
                self.hard_update(self.actor_local, self.actor_target)                     
                
    def soft_update(self,local,target,tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_params,local_params in zip(target.parameters(),local.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

    def hard_update(self,target,source):
        """Hard update model parameters.
        θ_target = θ_local
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_params,source_params in zip(target.parameters(),source.parameters()):
            target_params.data.copy_(source_params.data)    
            
class MADDPG():
    """
    Multi-Agent
    Wrapper class to manage multiple agents.
    """
    
    def __init__(self, num_agents, state_size, action_size):
        """Initialize MADDPGAgent object.
        Params
        ======
            num_agents (int): the number of agents in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        # create a shared replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, num_agents)
        # create a list with the num_agents agents
        self.agents = [DDPG_Agent(state_size, action_size, ii, num_agents, True, False)  for ii in range(num_agents)]
        
    def act(self,state):
        """Selects an action for each one of the agents in the list
        given their individual observations 
        and the current policy."""
        action = np.zeros([self.num_agents, self.action_size])
        for index,agent in enumerate(self.agents):
            action[index,:] = agent.act(state[index])
        return action
    
    def step(self,states,actions,rewards,next_states,dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states,actions,rewards,next_states,dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            for index,agent in enumerate(self.agents):
                agent.learn(experiences)
        
    def save_agents(self):
        """Save Weights of every agent"""
        for index,agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{index}.pth')
            torch.save(agent.critic_local.state_dict(),f'checkpoint_critic_{index}.pth')
            torch.save(agent.actor_target.state_dict(), f'checkpoint_actor_target_{index}.pth')
            torch.save(agent.critic_target.state_dict(),f'checkpoint_critic_target_{index}.pth')

    def load_agents(self):
        """Load Weights of every agent"""
        for index,agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load(f'checkpoint_actor_{index}.pth'))
            agent.critic_local.load_state_dict(torch.load(f'checkpoint_critic_{index}.pth'))
            agent.actor_target.load_state_dict(torch.load(f'checkpoint_actor_target_{index}.pth'))
            agent.critic_target.load_state_dict(torch.load(f'checkpoint_critic_target_{index}.pth'))           
            
class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.experience = namedtuple("Experience",field_names= ["states","actions","rewards","next_states","dones"])
        
    def add(self,state,action,reward,next_state,done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k = self.batch_size)
        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) 
                  for index in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) 
                   for index in range(self.num_agents)]
        rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) 
                   for index in range(self.num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) 
                       for index in range(self.num_agents)]
        dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
                 for index in range(self.num_agents)]
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)            