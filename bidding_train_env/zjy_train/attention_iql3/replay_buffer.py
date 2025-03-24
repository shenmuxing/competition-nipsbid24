import random
from collections import namedtuple
import numpy as np
import torch

Experience = namedtuple("Experience", field_names=["state_current_informations","state_history_informations", "action", "reward", 
                                                   "next_state_current_informations","next_state_history_informations", "done"])

class ReplayBuffer:
    """
    Reinforcement learning replay buffer for training data
    """

    def __init__(self):
        self.memory = []

    def push(self, state_current_informations, state_history_informations, 
             action, reward, next_state_current_informations, next_state_history_informations, done):
        """saving an experience tuple"""
        experience = Experience(state_current_informations,state_history_informations, action, reward, 
                                next_state_current_informations, next_state_history_informations, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """randomly sampling a batch of experiences"""
        tem = random.sample(self.memory, batch_size)
        state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones = zip(*tem)
        state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones = np.stack(state_current_informations),np.stack(state_history_informations), np.stack(actions), \
                                                    np.stack(rewards), np.stack(next_state_current_informations), np.stack(next_state_history_informations), np.stack(dones)
        state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones = torch.FloatTensor(state_current_informations), torch.FloatTensor(state_history_informations), torch.FloatTensor(actions),\
                                                    torch.FloatTensor(rewards), torch.FloatTensor(next_state_current_informations), torch.FloatTensor(next_state_history_informations), torch.FloatTensor(dones)
        return state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones

    def __len__(self):
        """return the length of replay buffer"""
        return len(self.memory)

if __name__ == '__main__':
    buffer = ReplayBuffer()
    for i in range(1000):
        buffer.push(np.array([1, 2, 3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    print(buffer.sample(20))
