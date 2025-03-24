# aiql.py: abbreviation for the attention-based IQL algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math
from copy import deepcopy
import os
from torch.distributions import Normal


class TimeEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TimeEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, pad the last dimension
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        # if time_step_index < self.pe.size(0):
        encoding = self.pe[:seq_len]  # [seq_len, 1, d_model]
        # else:
            # If time_step_index exceeds max_len, repeat the encoding
            # encoding = self.pe[-1].repeat(seq_len, 1, 1)
        return x + encoding

class Attention(nn.Module):
    def __init__(self, d_hist, d_curr, d_model, n_heads=8, dropout=0.1):
        """
        d_hist: Dimension of historical information
        d_curr: Dimension of current information
        d_model: Dimension of the model (output dimension)
        n_heads: Number of attention heads
        dropout: Dropout rate
        """
        super(Attention, self).__init__()
        self.d_model = d_model
        self.time_encoding = TimeEncoding(d_hist, max_len=48)

        # Define separate linear layers for Q, K, V
        self.linear_q = nn.Linear(d_curr, d_model, bias=False)
        self.linear_k = nn.Linear(d_hist, d_model, bias=False)
        self.linear_v = nn.Linear(d_hist, d_model, bias=False)

        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # LayerNorm layers
        self.norm1 = nn.LayerNorm(d_hist)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP for regression
        self.mlp = nn.Sequential(
            # nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # Assuming regression output is single-dimensional
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear_q.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_k.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_v.weight, nonlinearity='relu')
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_attention_vector(self, history_informations, current_informations):
        """
        history_informations: Tensor of shape [T, 1, D_hist]
        current_informations: Tensor of shape [1, 1, D_curr]
        """
        # Apply LayerNorm before attention (Pre-LN)

        history_informations = self.norm1(history_informations)
        # Apply time encoding
        history_encoded = self.time_encoding(history_informations)  # [T, 1, D_model]
        current_encoded = current_informations  # Assuming already encoded or no encoding needed

        # Project to Q, K, V
        Q = self.linear_q(current_encoded)  # [1, 1, D_model]
        K = self.linear_k(history_encoded)  # [T, 1, D_model]
        V = self.linear_v(history_encoded)  # [T, 1, D_model], 注意：T是时间的意思，所以是[L,N,D]而不是[n,L,D]的形式。

        # Apply LayerNorm before attention (Pre-LN)
        # Q = self.norm1(Q)
        # K = self.norm1(K)
        # V = self.norm1(V)

        # Compute attention
        attn_output, attn_weights = self.multihead_attn(Q, K, V)  # attn_output: [batch, seq, D_model]

        return attn_output  # [1,1,D_model]

    def forward(self, history_informations, current_informations):
        """
        history_informations: Tensor of shape [T, 1, D_hist]
        current_informations: Tensor of shape [1, 1, D_curr]
        time_step_index: int, current time step index for time encoding
        """
        attn_vector = self.get_attention_vector(history_informations , current_informations)  # [1,1,D_model]
        
        # Apply LayerNorm before MLP (Post-Attention LayerNorm with residual)
        attn_vector_norm = self.norm2(attn_vector)

        attn_vector_norm = F.relu(attn_vector_norm)

        return attn_vector_norm  # [1,1,D_model]

    def save_net(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        
        state_dict = {
            'time_encoding': self.time_encoding.state_dict(),
            'linear_q': self.linear_q.state_dict(),
            'linear_k': self.linear_k.state_dict(),
            'linear_v': self.linear_v.state_dict(),
            'multihead_attn': self.multihead_attn.state_dict(),
            'norm1': self.norm1.state_dict(),
            'norm2': self.norm2.state_dict(),
            'mlp': self.mlp.state_dict()
        }
        torch.save(state_dict, os.path.join(save_path, "attention_iql_model.pth"))

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        state_dict = torch.load(os.path.join(load_path, "attention_iql_model.pth"), map_location=device)
        
        self.time_encoding.load_state_dict(state_dict['time_encoding'])
        self.linear_q.load_state_dict(state_dict['linear_q'])
        self.linear_k.load_state_dict(state_dict['linear_k'])
        self.linear_v.load_state_dict(state_dict['linear_v'])
        self.multihead_attn.load_state_dict(state_dict['multihead_attn'])
        self.norm1.load_state_dict(state_dict['norm1'])
        self.norm2.load_state_dict(state_dict['norm2'])
        self.mlp.load_state_dict(state_dict['mlp'])

class Q(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(Q, self).__init__()
        self.dim_observation = num_of_states
        self.dim_action = 1  # 假设动作维度为1，如果不是，可以通过cfg进行设置

        self.obs_FC = nn.Linear(self.dim_observation, cfg.hidden_dim)
        # self.obs_BN = nn.BatchNorm1d(cfg.hidden_dim)
        self.action_FC = nn.Linear(self.dim_action, cfg.hidden_dim)
        # self.action_BN = nn.BatchNorm1d(cfg.hidden_dim)
        self.FC1 = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        # self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()

        for _ in range(cfg.num_hidden_layers - 1):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, acts):
        obs_embedding = F.relu(self.obs_FC(obs))
        action_embedding = F.relu(self.action_FC(acts))
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        x = F.relu(self.FC1(embedding))
        
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        
        q = self.FC_out(x)
        return q

class V(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(V, self).__init__()
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        # self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        # self.middle_bns = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        return self.FC_out(x)

class Actor(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(Actor, self).__init__()
        self.log_std_min = -10
        self.log_std_max = 2
        
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        # self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        # self.middle_bns = nn.ModuleList()
        for _ in range(cfg.num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_mu = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1
        self.FC_std = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        for layer in self.middle_layers:
            x = F.relu(layer(x))
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

class AIQL(nn.Module):
    def __init__(self, d_hist,d_curr,d_model, cfg):
        super().__init__()
        self.cfg = cfg

        self.attention = Attention(d_hist, d_curr, d_model, n_heads=8, dropout=0.1)

        num_of_states = d_curr+d_model  # 历史信息维度+当前信息维度
        self.num_of_states = num_of_states
        self.num_of_actions = 1  # Assuming action is 1-dimensional
        self.attention_lr = cfg.attention_lr
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
        # 定义单一优化器，并为不同的网络组件设置不同的学习率
        self.optimizer = Adam([
            {'params': self.attention.parameters(), 'lr': self.attention_lr},
            {'params': self.value_net.parameters(), 'lr': self.V_lr},
            {'params': self.actors.parameters(), 'lr': self.actor_lr},
            {'params': self.critic1.parameters(), 'lr': self.critic_lr},
            {'params': self.critic2.parameters(), 'lr': self.critic_lr},
        ])
        
        self.deterministic_action = False
        # self.use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        self.device = torch.device("cuda:1" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.critic1 = self.critic1.to(self.device)
            self.critic2 = self.critic2.to(self.device)
            self.critic1_target = self.critic1_target.to(self.device)
            self.critic2_target = self.critic2_target.to(self.device)
            self.value_net = self.value_net.to(self.device)
            self.actors = self.actors.to(self.device)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        self.max_grad_norm = cfg.max_grad_norm

    def step(self,state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones,\
            
            ):
        '''
        train model
        '''
        state_current_informations = state_current_informations.to(self.device)
        state_history_informations = state_history_informations.to(self.device)

        attention_vector = self.attention(state_history_informations, state_current_informations)
        states = torch.cat([state_current_informations, attention_vector], dim=-1).view(1,-1)
        
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        # next_states = next_states.to(self.device)
        next_state_current_informations = next_state_current_informations.to(self.device)
        next_state_history_informations = next_state_history_informations.to(self.device)
        with torch.no_grad():
            next_attention_vector = self.attention(next_state_history_informations, next_state_current_informations)
            next_states = torch.cat([next_state_current_informations, next_attention_vector], dim=-1).view(1,-1)
        dones = dones.to(self.device)

        # 计算各个损失
        value_loss = self.calc_value_loss(states, actions).mean()
        actor_loss = self.calc_policy_loss(states, actions).mean()
        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()

        # 累加所有损失以进行统一的反向传播
        # total_loss = value_loss + actor_loss + critic1_loss + critic2_loss
        # if 10*value_loss > torch.max(critic1_loss+critic2_loss, actor_loss):
            # total_loss = value_loss
        # else:
        total_loss = torch.max(critic1_loss+critic2_loss+value_loss, actor_loss)
        if torch.is_grad_enabled():
            # 清零优化器中的梯度
            self.optimizer.zero_grad()

            # assert not torch.isnan(total_loss).any()
            # 反向传播计算梯度
            total_loss.backward()

            # 进行梯度裁剪以防止梯度爆炸
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

            # 更新所有参数
            self.optimizer.step()

            # 更新目标网络
            self.update_target(self.critic1, self.critic1_target)
            self.update_target(self.critic2, self.critic2_target)

        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def forward(self, state_current_informations, state_history_informations):
        state_current_informations = state_current_informations.to(self.device)
        state_history_informations = state_history_informations.to(self.device)

        attention_vector = self.attention(state_history_informations, state_current_informations)
        states = torch.cat([state_current_informations, attention_vector], dim=-1).view(1,-1)
        
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

        self.attention.save_net(save_path)

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        state_dict = torch.load(load_path + "/iql_model.pth", map_location='cpu')
        
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.value_net.load_state_dict(state_dict['value_net'])
        self.actors.load_state_dict(state_dict['actors'])
        self.critic1_target.load_state_dict(state_dict['critic1_target'])
        self.critic2_target.load_state_dict(state_dict['critic2_target'])

        self.attention.load_net(load_path)
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
    model = AIQL(num_of_states, cfg)
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