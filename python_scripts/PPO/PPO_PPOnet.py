from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        super().__init__()
        self.node_num = node_num
        
        # 保留原有的特征提取网络结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 图神经网络部分
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        # --- 【核心修改 1】修改Actor头 ---
        # Actor不再输出一个离散概率，而是输出一个分布的参数
        # 1. mu_layer: 用于输出正态分布的均值(mu)
        # 2. log_sigma_layer: 用于输出log(sigma)，以保证sigma为正
        self.actor_mu = nn.Sequential(
            nn.Linear(200, act_dim),
            nn.Tanh()  # Tanh激活函数将mu的范围限制在[-1, 1]
        )
        
        # 将log_sigma作为可学习的参数，而不是依赖于状态。这是一种常见且稳定的做法。
        # act_dim 应该是动作的维度，这里是1
        self.actor_log_sigma = nn.Parameter(torch.zeros(1))
        
        # Critic头：输出状态值
        self.critic = nn.Linear(200, 1)
    
    # 保留原有的图处理函数
    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        # 特征提取部分与原DQN相同
        self.graph = self.creat_graph(x_graph)
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, dim=0)
        #x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        
        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))
        #print(f"State shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
        #print(f"State: {state}")
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        
        x_graph = self.creat_graph(x_graph)
        edge_index = x_graph.edge_index
        x_graph = self.conv_graph1(x_graph.x, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph2(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph3(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph4(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph5(x_graph, edge_index)
        x_graph = torch.mean(x_graph, dim=0)
        x_graph = self.fc_graph(x_graph)
        
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # Actor: 输出均值 mu
        mu = self.actor_mu(features)
        
        # 计算标准差 sigma
        # 使用exp来保证sigma是正数。广播log_sigma以匹配mu的批次大小
        log_sigma = self.actor_log_sigma.expand_as(mu)
        sigma = torch.exp(log_sigma)
        
        # 构建正态分布
        dist = Normal(mu, sigma)
        
        # Critic: 输出状态值 (保持不变)
        value = self.critic(features)
        
        return dist, value

class PPO:
    def __init__(self, node_num, env_information, act_dim=2):
        self.node_num = node_num
        self.env_information = env_information
        self.act_dim = act_dim
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.1  # PPO裁剪参数
        self.value_coef = 0.5  # 值函数损失系数
        self.entropy_coef = 0.01  # 增加熵系数，提高探索
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

        # 学习率和优化器参数
        self.lr = 3e-4  # 提高学习率
        self.lr_decay = 0.999  # 减缓学习率衰减

        # PPO更新参数
        self.update_epochs = 6  # 增加更新次数
        self.batch_size = 64  # 减小批大小，增加更新频率

        # 初始化策略网络
        self.policy = ActorCritic(act_dim=self.act_dim, node_num=self.node_num).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 存储轨迹数据
        self.states = []
        self.actions = []          # 合并存储动作 [shoulder, arm]
        self.rewards = []
        self.next_states = []
        self.values = []           # 合并存储价值
        self.log_probs = []       # 合并存储对数概率
        self.dones = []
    
    def choose_action(self, episode_num, obs, x_graph, action_type: str, explore=None):
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph


        # x = obs[0]
        # state = obs[1]

        # 确保输入张量在正确的设备上
        if isinstance(x, torch.Tensor):
            x = x.to(device)

        with torch.no_grad():
            # 策略网络输出动作分布的均值 mu 和价值 value
           # 策略网络输出一个2维的动作分布和一个价值
            dist, value = self.policy(x, state, x_graph) 
            
            # 从分布中采样一个2维的动作
            action_raw = dist.sample()
            # 对两个维度分别应用tanh
            action_scaled = torch.tanh(action_raw)
            
            # 计算对数概率（对两个维度的概率求和）
            action_raw_for_log_prob = torch.atanh(torch.clamp(action_scaled, -0.9999, 0.9999))
            log_prob = dist.log_prob(action_raw_for_log_prob).sum(dim=-1) # 这是关键点

            # 返回2维的动作向量、其总对数概率和状态值
            return action_scaled.cpu().numpy(), log_prob.item(), value.squeeze(-1).item()
    def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value_shoulder, value_arm, log_prob_shoulder, log_prob_arm):
        """
        存储轨迹数据
        动作值范围：[-1.0, 1.0]
        """
        # 将两个动作合并为一个 numpy 数组
        combined_action = np.array([action_shoulder, action_arm])
        # 将两个价值取平均值（或直接用一个），这里简单取第一个（因为它们理论上应该一样）
        combined_value = value_shoulder 
        
        self.states.append(state)
        self.actions.append(combined_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(combined_value)
        # 这里也存储 log_prob_shoulder，因为在 choose_action 里我们已经把两个log_prob加在一起了
        self.log_probs.append(log_prob_shoulder) 
        self.dones.append(done)
    
    def calculate_advantages(self, action_type: str):
        """
        计算优势函数和回报
        
        :param action_type: 'shoulder' 或 'arm'，指定计算哪个动作的优势函数
        """
        if not self.rewards:
            return np.array([]), np.array([])

        values = np.array(self.values) # 使用合并后的values
        
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        next_values = np.zeros_like(rewards, dtype=np.float32)
        with torch.no_grad():
            for idx in range(len(self.next_states)):
                if dones[idx]:
                    next_values[idx] = 0.0
                else:
                    x = self.next_states[idx][0]
                    state = self.next_states[idx][1]
                    x_graph = self.next_states[idx][2]
                    dist, v_next = self.policy(x, state, x_graph)
                    next_values[idx] = float(v_next.squeeze().item())

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def learn(self, action_type: str):
        """
        根据指定的动作类型（'shoulder' 或 'arm'）更新策略网络。

        :param action_type: 一个字符串，'shoulder' 或 'arm'，用于指定要更新哪个动作部分。
        """
        # 检查传入的参数是否合法
        advantages, returns = self.calculate_advantages(action_type)
        if len(advantages) == 0:
            return 0.0

        batch_states = self.states
        batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # --- 这是关键的修正 ---
        # 使用合并后的动作和对数概率
        batch_actions = torch.tensor(self.actions, dtype=torch.float32).to(device) # shape: [N, 2]
        batch_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(device) # shape: [N]
        
        total_loss = 0
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(batch_states))
            for start_idx in range(0, len(batch_states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                if not batch_indices.any(): # 空的批次跳过
                    continue

                # 处理当前批次的状态
                batch_x = [self.states[i][0] for i in batch_indices]
                batch_state = [self.states[i][1] for i in batch_indices]
                batch_x_graph = [self.states[i][2] for i in batch_indices]

                # 前向传播
                dist_list = []
                values_list = []
                for i in range(len(batch_x)):
                    dist, value = self.policy(batch_x[i], batch_state[i], batch_x_graph[i])
                    dist_list.append(dist)
                    values_list.append(value)
                
                # 将批次的结果堆叠起来
                # 修复维度问题：正确处理多维动作分布
                batch_mu = torch.cat([d.mean.unsqueeze(0) for d in dist_list], dim=0)
                batch_std = torch.cat([d.stddev.unsqueeze(0) for d in dist_list], dim=0)
                batch_dist = Normal(batch_mu, batch_std)
                values_batch = torch.cat(values_list).squeeze(-1)

                # 获取当前批次的数据
                batch_actions_curr = batch_actions[batch_indices] # shape: [M, 2]
                batch_log_probs_curr = batch_log_probs[batch_indices] # shape: [M]
                batch_advantages_curr = batch_advantages[batch_indices] # shape: [M]
                batch_returns_curr = batch_returns[batch_indices] # shape: [M]

                # 计算新的对数概率
                action_raw_batch = torch.atanh(torch.clamp(batch_actions_curr, -0.9999, 0.9999))
                new_log_probs = batch_dist.log_prob(action_raw_batch).sum(dim=-1) # 两个log prob求和
                entropy = batch_dist.entropy().mean()

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_log_probs_curr)
                surr1 = ratio * batch_advantages_curr
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages_curr
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_batch, batch_returns_curr)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        self.scheduler.step()

        # 清空轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("total_loss:", total_loss)
        return total_loss / self.update_epochs