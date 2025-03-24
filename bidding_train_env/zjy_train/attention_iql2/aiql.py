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
is_use_tensorboard = False

if is_use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
    # https://github.com/pytorch/pytorch/issues/91516
    # 当tensorboard出现问题时，可以考虑手动更改代码。

    
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

        # Initialize storage for variables to log
        self.last_Q = None
        self.last_K = None
        self.last_V = None
        self.last_attn_output = None

    def _init_weights(self):
        nn.init.kaiming_normal_(self.linear_q.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_k.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_v.weight, nonlinearity='relu')
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_attention_vector(self, history_informations, current_informations):
        """
        history_informations: Tensor of shape [T, N, D_hist]
        current_informations: Tensor of shape [1, N, D_curr]
        """
        # Apply LayerNorm before attention (Pre-LN)

        history_informations = self.norm1(history_informations)
        # Apply time encoding
        history_encoded = self.time_encoding(history_informations)  # [T, N, D_model]
        current_encoded = current_informations  # Assuming already encoded or no encoding needed
        # current_encoded = torch.ones(1, 1, self.d_model).to(history_informations.device) 
        # Project to Q, K, V
        Q = self.linear_q(current_encoded)  # [1, N, D_model]
        # Q = current_encoded
        K = self.linear_k(history_encoded)  # [T, N, D_model]
        V = self.linear_v(history_encoded)  # [T, N, D_model], 注意：T是时间的意思，所以是[L,N,D]而不是[n,L,D]的形式。

        if is_use_tensorboard:
            # Store variables for TensorBoard
            self.last_Q = Q.detach().cpu()
            self.last_K = K.detach().cpu()
            self.last_V = V.detach().cpu()

        # Compute attention
        attn_output, attn_weights = self.multihead_attn(Q, K, V)  # attn_output: [batch, seq, D_model]
        if is_use_tensorboard:
            self.last_attn_output = attn_output.detach().cpu()
        
        return attn_output  # [1,N,D_model]

    def forward(self, history_informations, current_informations):
        """
        history_informations: Tensor of shape [T, 1, D_hist]
        current_informations: Tensor of shape [1, 1, D_curr]
        """
        attn_vector = self.get_attention_vector(history_informations , current_informations)  # [1,1,D_model]
        
        # Apply LayerNorm before MLP (Post-Attention LayerNorm with residual)
        attn_vector_norm = self.norm2(attn_vector)

        attn_vector_norm = F.leaky_relu(attn_vector_norm)

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

        self.obs_FC_k = nn.Linear(self.dim_observation, cfg.hidden_dim)
        self.obs_FC_v = nn.Linear(self.dim_observation, cfg.hidden_dim)

        self.action_FC = nn.Linear(self.dim_action, cfg.hidden_dim)
        self.FC1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(cfg.hidden_dim, 8, dropout=0.1, batch_first=False)
        
        self.middle_layers = nn.ModuleList()

        for _ in range(cfg.critic_num_hidden_layers - 1):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self.FC_leaky_relu = nn.LeakyReLU(0.05)

        self._init_weights()

        # Initialize storage for variables to log
        self.last_obs_embedding = None
        self.last_action_embedding = None
        self.last_q = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, acts):
        # obs_embedding = F.relu(self.obs_FC(obs)) # key, value
        # action_embedding = F.relu(self.action_FC(acts)) # query
        obs_embedding_k = self.obs_FC_k(obs).view(1,1,-1) # key,
        obs_embedding_v = self.obs_FC_v(obs).view(1,1,-1) #value
        action_embedding = F.relu(self.action_FC(acts).view(1,1,-1)) # query
        embedding,_ = self.multihead_attn(action_embedding, obs_embedding_k, obs_embedding_v)

        embedding = embedding.view(1,-1)
        # embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        x = F.relu(self.FC1(embedding)) 
        
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        
        x = x + embedding  # residual connection

        q = self.FC_out(x)
        q = self.FC_leaky_relu(q)

        if is_use_tensorboard:
            # Store variables for TensorBoard
            self.last_obs_embedding = obs_embedding_k.detach().cpu()
            self.last_action_embedding = action_embedding.detach().cpu()
            self.last_q = q.detach().cpu()

        return q

class V(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(V, self).__init__()
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        # self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        # self.middle_bns = nn.ModuleList()
        for _ in range(cfg.critic_num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_out = nn.Linear(cfg.hidden_dim, 1)

        self.FC_leaky_relu = nn.LeakyReLU(0.05)

        self._init_weights()

        # Initialize storage for variables to log
        self.last_obs_embedding = None
        self.last_value = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        obs_embedding = F.relu(self.FC1(obs))
        x = obs_embedding
        for layer in self.middle_layers:
            x = F.relu(layer(x))

        value = self.FC_out(x)
        value = self.FC_leaky_relu(value)
        
        if is_use_tensorboard:
            # Store variables for TensorBoard
            self.last_obs_embedding = obs_embedding.detach().cpu()
            self.last_value = value.detach().cpu()

        return value

class Actor(nn.Module):
    def __init__(self, num_of_states, cfg):
        super(Actor, self).__init__()
        self.log_std_min = -10
        self.log_std_max = 2.7
        
        self.FC1 = nn.Linear(num_of_states, cfg.hidden_dim)
        # self.BN1 = nn.BatchNorm1d(cfg.hidden_dim)
        
        self.middle_layers = nn.ModuleList()
        # self.middle_bns = nn.ModuleList()
        for _ in range(cfg.actor_num_hidden_layers):
            self.middle_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            # self.middle_bns.append(nn.BatchNorm1d(cfg.hidden_dim))
        
        self.FC_mu = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1
        self.FC_std = nn.Linear(cfg.hidden_dim, 1)  # 假设动作维度为1

        self._init_weights()

        # Initialize storage for variables to log
        self.last_mu = None
        self.last_log_std = None
        self.last_obs_embedding = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        obs_embedding = F.relu(self.FC1(obs))

        if is_use_tensorboard:
            self.last_obs_embedding = obs_embedding.detach().cpu()
    
        x = obs_embedding
        for layer in self.middle_layers:
            x = F.relu(layer(x))
        
        x = x + obs_embedding  # residual connection
        mu = self.FC_mu(x)
        log_std = self.FC_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        if is_use_tensorboard:
            self.last_mu = mu.detach().cpu()
            self.last_log_std = log_std.detach().cpu()

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

        num_of_states_full = d_curr+d_model  +cfg.D_other  # 历史信息维度+当前信息维度 + 其它维度
        num_of_observations = d_curr+d_model  # 当前信息维度+历史信息维度
        self.num_of_states_full = num_of_states_full
        self.num_of_observations = num_of_observations
        self.num_of_actions = 1  # Assuming action is 1-dimensional
        self.attention_lr = cfg.attention_lr
        self.V_lr = cfg.V_lr
        self.critic_lr = cfg.critic_lr
        self.actor_lr = cfg.actor_lr
        self.network_random_seed = cfg.network_random_seed
        self.expectile = cfg.expectile
        self.temperature = cfg.temperature
        torch.random.manual_seed(self.network_random_seed)
        self.value_net = V(self.num_of_states_full, cfg)
        self.critic1 = Q(self.num_of_states_full, cfg)
        self.critic2 = Q(self.num_of_states_full, cfg)
        self.critic1_target = Q(self.num_of_states_full, cfg)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(self.num_of_states_full, cfg)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actors = Actor(num_of_observations, cfg)
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

        self.use_cuda = cfg.use_cuda
        self.device = torch.device("cuda:1" if self.use_cuda and torch.cuda.is_available() else "cpu")
        if self.use_cuda:
            self.critic1 = self.critic1.to(self.device)
            self.critic2 = self.critic2.to(self.device)
            self.critic1_target = self.critic1_target.to(self.device)
            self.critic2_target = self.critic2_target.to(self.device)
            self.value_net = self.value_net.to(self.device)
            self.actors = self.actors.to(self.device)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        
        self.max_grad_norm = cfg.max_grad_norm

        # Initialize TensorBoard SummaryWriter if enabled
        if is_use_tensorboard:
            self.writer = SummaryWriter()
        else:
            self.writer = None
        self.global_step = 0  # To keep track of steps for logging

        # 新增：用于存储梯度信息的字典
        self.grad_dict = {}

    def register_hooks(self):
        def hook_fn(name):
            def fn(grad):
                self.grad_dict[name] = grad.detach().cpu().numpy()
            return fn

        # 为每个主要组件注册钩子
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(hook_fn(name))

    def step(self,state_current_informations, state_history_informations, actions, rewards, \
            next_state_current_informations, next_state_history_informations, dones,
            state_current_informations_other = None, next_state_current_informations_other = None,
            mode = 'all'):
        '''
        train model
        mode: 'all' train all networks, 
              'actor' only train actor network, 
              'critic' only train critic network
        '''
        state_current_informations = state_current_informations.to(self.device)
        state_history_informations = state_history_informations.to(self.device)

        attention_vector = self.attention(state_history_informations, state_current_informations)
        if state_current_informations_other is not None:
            states = torch.cat([state_current_informations, attention_vector,state_current_informations_other], dim=-1).view(-1,self.num_of_states_full)
            # states = torch.cat([state_current_informations, attention_vector], dim=-1).view(1,-1)
            observations = torch.cat([state_current_informations, attention_vector.detach()], dim=-1).view(-1,self.num_of_observations)
        else:
            states = torch.cat([state_current_informations, attention_vector], dim=-1).view(-1,self.num_of_states_full)
            observations = torch.cat([state_current_informations, attention_vector.detach()], dim=-1).view(-1,self.num_of_observations)
        
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        # next_states = next_states.to(self.device)
        next_state_current_informations = next_state_current_informations.to(self.device)
        next_state_history_informations = next_state_history_informations.to(self.device)
        with torch.no_grad():
            next_attention_vector = self.attention(next_state_history_informations, next_state_current_informations)
            if next_state_current_informations_other is not None:
                next_states = torch.cat([next_state_current_informations, next_attention_vector,next_state_current_informations_other], dim=-1).view(-1,self.num_of_states_full)
                # next_states = torch.cat([next_state_current_informations, next_attention_vector], dim=-1).view(1,-1)
                # next_observations = torch.cat([next_state_current_informations, next_attention_vector.detach()], dim=-1).view(-1,self.num_of_observations)
            else:
                next_states = torch.cat([next_state_current_informations, next_attention_vector], dim=-1).view(-1,self.num_of_states_full)
                # next_observations = torch.cat([next_state_current_informations, next_attention_vector.detach()], dim=-1).view(-1,self.num_of_observations)
        
        dones = dones.to(self.device)

        if mode == 'all' or mode == 'actor':
            actor_loss = self.calc_policy_loss(states,observations, actions, rewards, dones, next_states).mean()
            value_loss = actor_loss.detach()
            critic1_loss,critic2_loss = actor_loss.detach(),actor_loss.detach()
        else:
            # 计算各个损失
            value_loss = self.calc_value_loss(states, actions,rewards,dones,next_states).mean()
            critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)

            actor_loss = critic1_loss.detach()

        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()

        if (mode == 'actor' or mode == 'all') and actor_loss > 200 and False:
            with torch.no_grad():
                v = self.value_net(states)
                next_v = self.value_net(next_states)
                q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

            exp_a = torch.exp((q_target - v) * self.temperature)
            exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(self.device))

            _, dist = self.actors.evaluate(observations)
            log_probs = dist.log_prob(actions)
            print(f"rewards:{rewards},actions:{actions},value_loss:{value_loss},actor_loss:{actor_loss}")
            print(f"q_target: {q_target},v: {v},exp_a: {exp_a}")
            print(f"dist.log_prob(actions): {log_probs}")

        # total_loss = torch.max(critic1_loss+critic2_loss+value_loss, actor_loss)
        if mode == 'all':
            total_loss = critic1_loss+critic2_loss+value_loss+actor_loss
        elif mode == 'actor':
            total_loss = actor_loss
        elif mode == 'critic':
            total_loss = critic1_loss + critic2_loss + value_loss

        if torch.is_grad_enabled():
            # 清零优化器中的梯度
            self.optimizer.zero_grad()

            if is_use_tensorboard:
                # 清空梯度字典
                self.grad_dict.clear()
                # 注册钩子
                self.register_hooks()

            # 反向传播计算梯度
            total_loss.backward()

            # 进行梯度裁剪以防止梯度爆炸
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

            # 更新所有参数
            self.optimizer.step()

            # 更新目标网络
            self.update_target(self.critic1, self.critic1_target)
            self.update_target(self.critic2, self.critic2_target)

        # Logging with TensorBoard
        if self.writer:
            def safe_add_histogram(name, values, step):
                if values is not None:
                    if isinstance(values, torch.Tensor):
                        values = values.detach().cpu().numpy()
                    if isinstance(values, np.ndarray):
                        if values.ndim == 2 and values.shape[0] == 1:
                            values = values.squeeze(0)  # 从 [1, N] 压缩到 [N]
                        if values.size > 0:
                            self.writer.add_histogram(name, values, step)

            # Attention
            safe_add_histogram('Attention/Q', self.attention.last_Q, self.global_step)
            safe_add_histogram('Attention/K', self.attention.last_K, self.global_step)
            safe_add_histogram('Attention/V', self.attention.last_V, self.global_step)
            # add_embedding(self.attention.last_K, 'Attention/K')
            # add_embedding(self.attention.last_V, 'Attention/V')

            safe_add_histogram('Attention/attention_output', self.attention.last_attn_output, self.global_step)

            # Q Network (using critic1 for logging)
            safe_add_histogram('Q/obs_embedding', self.critic1.last_obs_embedding, self.global_step)
            safe_add_histogram('Q/action_embedding', self.critic1.last_action_embedding, self.global_step)
            safe_add_histogram('Q/q', self.critic1.last_q, self.global_step)

            # Value Network
            safe_add_histogram('Value/obs_embedding', self.value_net.last_obs_embedding, self.global_step)
            safe_add_histogram('Value/value', self.value_net.last_value, self.global_step)

            # Actor
            safe_add_histogram('Actor/mu', self.actors.last_mu, self.global_step)
            safe_add_histogram('Actor/log_std', self.actors.last_log_std, self.global_step)

            # AIQL
            safe_add_histogram('AIQL/states', states.detach().cpu(), self.global_step)
            safe_add_histogram('AIQL/observations', observations.detach().cpu(), self.global_step)

            # 新增：记录梯度信息
            for name, grad in self.grad_dict.items():
                self.writer.add_histogram(f'Gradients/{name}', grad, self.global_step)

            # Log additional distributions if needed
            mu = self.actors.last_mu
            log_std = self.actors.last_log_std
            if mu is not None and log_std is not None:
                std = torch.exp(torch.tensor(log_std))
                safe_add_histogram('AIQL/dist_mu', mu, self.global_step)
                safe_add_histogram('AIQL/dist_std', std, self.global_step)

            # Optional: Log losses as scalars (这些不需要更改，因为它们已经是标量)
            self.writer.add_scalar('Loss/Critic1_Loss', critic1_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Critic2_Loss', critic2_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Value_Loss', value_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Actor_Loss', actor_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Total_Loss', total_loss.item(), self.global_step)

            # Increment global step
            self.global_step += 1

        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def forward(self, state_current_informations, state_history_informations):
        state_current_informations = state_current_informations.to(self.device)
        state_history_informations = state_history_informations.to(self.device)

        attention_vector = self.attention(state_history_informations, state_current_informations)
        observations = torch.cat([state_current_informations, attention_vector], dim=-1).view(1,-1)
        
        actions = self.actors.get_det_action(observations)
        actions = torch.clamp(actions, min=0)
        return actions

    def calc_policy_loss(self, states,observations, actions,rewards, dones, next_states):
        with torch.no_grad():
            v = self.value_net(states)
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        exp_a = torch.exp((q_target - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100]).to(self.device))
        # exp_a = F.sigmoid((q_target - v)*self.temperature)

        _, dist = self.actors.evaluate(observations)
        log_probs = dist.log_prob(actions)
        probs = F.sigmoid(log_probs)
        actor_loss =  1-(exp_a * probs).mean()
        # actor_loss = - (exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, states, actions,rewards,dones,next_states):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            # q2 = self.critic2_target(states, actions)
            # q2 = q1.detach() + 100
            min_Q = q1
            # q1_target = rewards+ (self.GAMMA * (1 - dones) * self.value_net(next_states))
            # min_Q = q1_target

        value = self.value_net(states)
        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()

        # value_now = value.clone().detach()
        # next_value = self.value_net(next_states)
        # return_value = rewards + (self.GAMMA * (1 - dones) * next_value)
        # exp_a = torch.clamp(torch.exp((return_value - value_now).clone().detach()),0,50).mean()
        # exp_a = F.sigmoid(return_value - value_now).mean()
        # value_loss += exp_a * 0.01 * F.mse_loss(return_value, value_now,reduce=True) # 一个正则项，使得value网络的输出接近实际的return值

        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        q1 = self.critic1(states, actions)
        # q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        # critic2_loss = ((q2 - q_target) ** 2).mean()
        critic2_loss = critic1_loss.detach() * 0
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

    def close_writer(self):
        if self.writer:
            self.writer.close()
