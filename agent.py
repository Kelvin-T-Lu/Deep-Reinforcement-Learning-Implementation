import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            ### CODE #### 
            # Choose a random action
            action = torch.tensor([[random.randrange(self.action_size)]],
                             device=device, dtype=torch.long)

        else:
            ### CODE ####
            # Choose the best action
            action = self.policy_net(torch.from_numpy(state)).max(1)[1].view(1, 1)
        return action

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        musk = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)


        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        state_action_values = self.policy_net(
            states).gather(1, actions.unsqueeze(1))

        # Compute Q function of next state
        ### CODE ####
        next_states = torch.from_numpy(next_states).cuda()
        non_final_next_states = next_states[musk == 1]
        net_outputs = self.policy_net(non_final_next_states)

        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        next_states_value = torch.zeros(batch_size).cuda()
        next_states_value[musk == 1] = net_outputs.max(1)[0].detach()
        # Compute the Huber Loss
        ### CODE ####
        expected_state_action_values = rewards + \
            self.discount_factor * next_states_value

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


