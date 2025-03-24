# iql.py
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import torch
from copy import deepcopy
import os
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Q(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(Q, self).__init__()
        self.dim_observation = num_of_states
        self.dim_action = 1  # 假设动作维度为1，如果不是，可以通过cfg进行设置

        self.obs_FC = nn.Linear(self.dim_observation, cfg.hidden_dim)
        self.obs_BN = nn.BatchNorm1d(cfg.hidden_dim)
        self.action_FC = nn.Linear(self.dim_action, cfg.hidden_dim)
        self.action_BN = nn.BatchNorm1d(cfg.hidden_dim)
        self.FC1 = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        self.middle_bns = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers - 1):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, acts):
        obs_embedding = F.relu(self.obs_BN(self.obs_FC(obs)))
        action_embedding = F.relu(self.action_BN(self.action_FC(acts)))
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        x = F.relu(self.BN1(self.FC1(embedding)))
        
        for layer, bn in zip(self.middle_layers, self.middle_bns):
            x = F.relu(bn(layer(x)))
        
        q = self.FC_out(x)
        return q

class V(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(V, self).__init__()
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        self.middle_bns = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = F.relu(self.BN1(self.FC1(obs)))
        for layer, bn in zip(self.middle_layers, self.middle_bns):
            x = F.relu(bn(layer(x)))
        return self.FC_out(x)

class Actor(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(Actor, self).__init__()
        self.log_std_min = -10
        self.log_std_max = 2
        
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        self.middle_bns = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_mu = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1
        self.FC_std = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = F.relu(self.BN1(self.FC1(obs)))
        for layer, bn in zip(self.middle_layers, self.middle_bns):
            x = F.relu(bn(layer(x)))
        mu = self.FC_mu(x)
        log_std = self.FC_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, obs, epsilon=1e-6):
        # 此函数要求obs必须不含nan
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()

    def get_det_action(self, obs):
        mu, _ = self.forward(obs)
        return mu.detach().cpu()

class IQL(nn.Module):
    def __init__(self, num_of_states, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_of_states = num_of_states
        self.num_of_actions = 1  # Assuming action is 1-dimensional
        self.V_lr = cfg.V_lr
        self.critic_lr = cfg.critic_lr
        self.actor_lr = cfg.actor_lr
        self.network_random_seed = cfg.network_random_seed
        self.expectile = cfg.expectile
        self.temperature = cfg.temperature
        torch.random.manual_seed(self.network_random_seed)
        self.value_net = V(num_of_states, cfg)
        self.critic1 = Q(num_of_states, cfg)
        self.critic2 = Q(num_of_states, cfg)
        self.critic1_target = Q(num_of_states, cfg)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(num_of_states, cfg)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actors = Actor(num_of_states, cfg)
        self.GAMMA = cfg.gamma
        self.tau = cfg.tau
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
        self.deterministic_action = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.critic1.cuda()
            self.critic2.cuda()
            self.critic1_target.cuda()
            self.critic2_target.cuda()
            self.value_net.cuda()
            self.actors.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        self.max_grad_norm = cfg.max_grad_norm

    def step(self, states, actions, rewards, next_states, dones, weights):
        '''
        train model
        '''
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # Modify optimizers' learning rates
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = self.V_lr * weights.mean().item()
        for param_group in self.critic1_optimizer.param_groups:
            param_group['lr'] = self.critic_lr * weights.mean().item()
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = self.critic_lr * weights.mean().item()
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_lr * weights.mean().item()

        # Value network update
        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss = (value_loss * weights).mean()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
        self.value_optimizer.step()

        # Actor network update
        actor_loss = self.calc_policy_loss(states, actions)
        actor_loss = (actor_loss * weights).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()

        # Critic networks update
        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)
        critic1_loss = (critic1_loss * weights).mean()
        critic2_loss = (critic2_loss * weights).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.max_grad_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.max_grad_norm)
        self.critic2_optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()


    def take_actions(self, states):
        '''
        take action
        '''
        states = torch.Tensor(states).type(self.FloatTensor)
        if self.deterministic_action:
            actions = self.actors.get_det_action(states)
        else:
            actions = self.actors.get_action(states)
        actions = torch.clamp(actions, 0)
        actions = actions.cpu().data.numpy()
        return actions

    def forward(self, states):

        actions = self.actors.get_det_action(states)
        actions = torch.clamp(actions, min=0)
        return actions

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2)

        exp_a = torch.exp(min_Q - v) * self.temperature
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device))

        _, dist = self.actors.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2)

        value = self.value_net(states)
        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def save_net(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        state_dict = {
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'value_net': self.value_net.state_dict(),
            'actors': self.actors.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict()
        }
        torch.save(state_dict, os.path.join(save_path, "iql_model.pth"))

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        state_dict = torch.load(load_path + "/iql_model.pth", map_location='cpu')
        
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.value_net.load_state_dict(state_dict['value_net'])
        self.actors.load_state_dict(state_dict['actors'])
        self.critic1_target.load_state_dict(state_dict['critic1_target'])
        self.critic2_target.load_state_dict(state_dict['critic2_target'])

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

if __name__ == '__main__':
    # You'll need to define a Config class or use a dictionary for cfg
    class Config:
        def __init__(self):
            self.hidden_dim = 256
            self.num_hidden_layers = 2
            self.V_lr = 0.001
            self.critic_lr = 0.001
            self.actor_lr = 0.001
            self.network_random_seed = 42
            self.expectile = 0.8
            self.temperature = 0.1
            self.gamma = 0.99
            self.tau = 0.005

    cfg = Config()
    num_of_states = 3  # Assuming 3 state dimensions
    model = IQL(num_of_states, cfg)
    step_num = 100
    batch_size = 1000
    for i in range(step_num):
        states = np.random.uniform(2, 5, size=(batch_size, num_of_states))
        next_states = np.random.uniform(2, 5, size=(batch_size, num_of_states))
        actions = np.random.uniform(-1, 1, size=(batch_size, 1))
        rewards = np.random.uniform(0, 1, size=(batch_size, 1))
        terminals = np.zeros((batch_size, 1))
        states, next_states, actions, rewards, terminals = torch.tensor(states, dtype=torch.float), torch.tensor(
            next_states, dtype=torch.float), torch.tensor(actions, dtype=torch.float), torch.tensor(rewards,
                                                                                                    dtype=torch.float), torch.tensor(
            terminals, dtype=torch.float)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        print(f'step:{i} q_loss:{q_loss} v_loss:{v_loss} a_loss:{a_loss}')