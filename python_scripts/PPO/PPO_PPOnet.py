from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
from torch.distributions import Normal


class LMFModule(nn.Module):
    """
    低秩多模态融合模块：
    - 用于融合图像特征 x 和状态特征 state
    - 与 lmfGrasp 中的 LMFModule 保持一致，便于论文/代码对应
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim

        self.fc_x_list = nn.ModuleList([
            nn.Linear(input_dim1, hidden_dim) for _ in range(rank)
        ])
        self.fc_s_list = nn.ModuleList([
            nn.Linear(input_dim2, hidden_dim) for _ in range(rank)
        ])

        self.fc_fusion = nn.Linear(rank * hidden_dim, hidden_dim)

    def forward(self, x, state):
        """
        x:     [B, input_dim1]
        state: [B, input_dim2]
        返回:  [B, hidden_dim]
        """
        batch_size = x.size(0)
        fusion_tensor = torch.zeros(batch_size, self.rank, self.hidden_dim, device=x.device)

        for i in range(self.rank):
            x_proj = self.fc_x_list[i](x)
            s_proj = self.fc_s_list[i](state)
            # Hadamard product
            fusion_tensor[:, i, :] = x_proj * s_proj

        fusion_flat = fusion_tensor.view(batch_size, -1)
        fused = self.fc_fusion(fusion_flat)
        return fused


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

        # 图像特征 + 状态特征 的 LMF 多模态融合模块
        # 与 lmfGrasp 中保持相同配置：100 + 100 -> 200
        self.lmf = LMFModule(input_dim1=100, input_dim2=100, hidden_dim=200, rank=5)
        
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
        # 【修复】降低初始探索噪声：从0（对应sigma=1.0）改为-1.0（对应sigma≈0.37）
        # 这样可以减少过度探索，让网络更快收敛到好的策略
        self.actor_log_sigma = nn.Parameter(torch.tensor([-1.0]))  # 初始sigma ≈ 0.37，而不是1.0
        
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

        # 使用 LMF 融合图像特征和状态特征
        img_feat = normalized_data1.unsqueeze(0)   # [1, 100]
        state_feat = normalized_data2.unsqueeze(0) # [1, 100]
        fused_feat = self.lmf(img_feat, state_feat).squeeze(0)  # [200]

        # 再与图特征拼接，得到最终 300 维特征
        state_x = torch.cat((fused_feat, normalized_x_graph), dim=-1)
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
    def __init__(self, node_num, env_information, act_dim=1):
        self.node_num = node_num
        self.env_information = env_information
        self.act_dim = act_dim
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.1  # PPO裁剪参数
        self.value_coef = 0.01  # 【修复】大幅降低值函数损失系数，让reward对loss的影响更大（从0.1降低到0.01，目标：Reward=-90时loss≈7~9）
        self.entropy_coef = 0.01  # 【优化】降低熵系数，减少探索，提高收敛稳定性（从0.1降低到0.01）
        self.policy_loss_scale = 0.1  # 【新增】policy_loss缩放因子，控制loss大小（目标：Reward=-90时loss≈7~9）
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

        # 学习率和优化器参数
        self.lr = 3e-4  # 提高学习率
        self.lr_decay = 0.999  # 减缓学习率衰减

        # PPO更新参数
        self.update_epochs = 8  # 【优化】增加更新次数，提高学习稳定性（从4增加到8）
        self.batch_size = 64  # 批大小
        # 【新增】保留最近若干个 episode 进行学习（这里设为最近10个）
        # 注意：仍然是 on-policy，只是把最近几轮的数据一起打包成更大的 batch
        self.max_episode_buffer = 10
        self.episode_buffer = deque(maxlen=self.max_episode_buffer)

        # 初始化策略网络
        self.policy = ActorCritic(act_dim=self.act_dim, node_num=self.node_num).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 当前 episode 的轨迹数据（每轮 episode 结束后会被打包进 episode_buffer）
        self.states = []
        self.actions = []          # 存储 tanh 后的动作（用于环境执行）
        self.actions_raw = []      # 存储原始动作（用于正确计算 log_prob）
        self.rewards = []
        self.next_states = []
        self.values = []           # 合并存储价值
        self.log_probs = []       # 合并存储对数概率
        self.dones = []
    
    def choose_action(self, episode_num, obs, x_graph, explore=None):
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph

        if isinstance(x, torch.Tensor):
            x = x.to(device)

        epsilon = max(0.1, 0.90 - episode_num * 0.0001)
        if explore is not None:
            use_random = explore
        else:
            random_num = np.random.uniform()
            use_random = random_num < epsilon
            
        with torch.no_grad():
            # 关键修改：不再使用 action_type 来区分生成逻辑
            # 直接从 policy 网络获得分布
            dist, value = self.policy(x, state, x_graph)
            
            # 探索或利用的逻辑现在基于 self.act_dim
            if use_random:
                # 探索：根据智能体的动作维度生成随机动作
                action_scaled = torch.tensor(np.random.uniform(-1, 1, size=self.act_dim), dtype=torch.float32).to(device)
                # 对于随机动作，需要从分布中采样一个 action_raw 来计算 log_prob
                # 但为了简化，我们使用一个近似：从分布中采样，然后 tanh
                action_raw = dist.sample()
                action_scaled = torch.tanh(action_raw)
            else:
                # 利用：从策略网络生成的分布中采样
                action_raw = dist.sample()
                action_scaled = torch.tanh(action_raw)
            
            # 【关键修复】正确计算 log_prob：需要减去 tanh 的雅可比修正项
            # log_prob = dist.log_prob(action_raw) - log(1 - tanh²(action_raw))
            # 这是 tanh squashing 的标准修正公式
            log_prob_raw = dist.log_prob(action_raw)
            # 计算 tanh 的雅可比行列式修正项：log(1 - tanh²(x))
            tanh_correction = torch.log(1 - action_scaled.pow(2) + 1e-6)  # 添加小值防止 log(0)
            log_prob = (log_prob_raw - tanh_correction).sum(dim=-1)

            # 返回一个标量动作、一个标量概率、一个标量价值、原始动作
            # 注意：即使 self.act_dim > 1，这里也直接返回张量，由调用者处理
            # 但你的情况是 act_dim=1, 所以返回的就是一个标量张量
            return action_scaled.cpu().numpy(), log_prob.item(), value.item(), action_raw.cpu().numpy()

    def store_transition_catch(self, state, action, reward, next_state, done, value, log_prob, action_raw=None):
        """
        --- 【修改 3】简化存储接口 ---
        每个智能体只存储自己的数据。
        
        参数:
            action: tanh 后的动作（用于环境执行）
            action_raw: 原始动作（用于正确计算 log_prob），如果为 None 则从 action 反推
        """
        self.states.append(state)
        self.actions.append(action)
        # 如果提供了 action_raw，存储它；否则尝试从 action 反推（但这不是最优的）
        if action_raw is not None:
            self.actions_raw.append(action_raw)
        else:
            # 尝试从 tanh 后的动作反推原始动作（用于兼容性）
            # 注意：这不是最优方法，但为了向后兼容
            action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
            action_raw_approx = torch.atanh(torch.clamp(action_tensor, -0.9999, 0.9999))
            self.actions_raw.append(action_raw_approx.cpu().item())
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def calculate_advantages(self, rewards, values, dones):
        """
        计算优势函数和回报
        """
        if not rewards:          # 没有数据直接返回空
            return np.array([]), np.array([])

        # 将rewards和values转换为numpy数组以便处理
        if len(values) != len(rewards):
            print(f"警告: values 长度 ({len(values)}) 和 rewards 长度 ({len(rewards)}) 不匹配！这可能表明数据存储逻辑有误。")
            # 可以选择报错，或者截断到较短的那个长度（不推荐）
            # 这里选择报错，让开发者定位问题
            raise ValueError("Critical Error: self.values and self.rewards have different lengths.")

        values = np.array(values) 
        rewards = np.array(rewards)
        dones = np.array(dones)

        # 计算GAE优势函数
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        # 从后向前计算优势函数
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 对于最后一个时间步，使用0作为下一个值的估计
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # 计算回报
        returns = advantages + values

        # 【新增】记录reward和advantages的统计信息，帮助诊断问题
        rewards_sum = rewards.sum()
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        advantages_mean_before = advantages.mean()
        advantages_std_before = advantages.std()
        
        # 【关键修复】改进优势函数标准化，避免过度削弱负奖励的影响
        # 问题：原来的标准化会将所有优势函数标准化为均值0、标准差1
        # 这会导致即使奖励是-500，标准化后可能只是-1或-2，网络无法区分"非常坏"和"一般坏"
        # 
        # 解决方案：使用更温和的标准化，保留负奖励的相对强度
        # 方法1：只中心化（减去均值），不除以标准差（保留原始尺度）
        # 方法2：使用更温和的标准化（除以标准差，但保留更多原始信息）
        # 这里使用方法1：只中心化，保留原始尺度，让负奖励的影响更明显
        advantages_mean = advantages.mean()
        advantages = advantages - advantages_mean  # 只中心化，不标准化
        
        # 【新增】如果优势函数方差太大，适度缩放（但不完全标准化），保留更多reward信息
        # 【修复】提高缩放阈值，让advantages保持更大的数值，使loss对reward更敏感（目标：Reward=-90时loss≈7~9）
        advantages_std = advantages.std()
        if advantages_std > 200.0:  # 【修复】提高阈值从50.0到200.0，保留更多reward信息
            # 缩放因子：将标准差压缩到200以内，但保留相对大小
            scale_factor = 200.0 / (advantages_std + 1e-8)
            advantages = advantages * scale_factor
            print(f"  【优势函数缩放】原始std={advantages_std:.2f}, 缩放因子={scale_factor:.4f}, 缩放后std={advantages.std():.2f}")
        
        # 【新增】打印reward和advantages的统计信息（每10个episode打印一次，避免输出过多）
        # 注意：这里无法直接获取episode_num，所以每次都打印，但可以通过外部控制
        advantages_mean_after = advantages.mean()
        advantages_std_after = advantages.std()
        print(f"  【Reward统计】sum={rewards_sum:.2f}, mean={rewards_mean:.2f}, std={rewards_std:.2f}")
        print(f"  【Advantages统计】标准化前: mean={advantages_mean_before:.2f}, std={advantages_std_before:.2f}")
        print(f"  【Advantages统计】标准化后: mean={advantages_mean_after:.2f}, std={advantages_std_after:.2f}")

        return advantages, returns

    def get_current_sigma(self):
        return torch.exp(self.policy.actor_log_sigma).item()
    def learn(self):
        """
        根据指定的动作类型（'shoulder' 或 'arm'）更新策略网络。
  
        :param action_type: 一个字符串，'shoulder' 或 'arm'，用于指定要更新哪个动作部分。
        """
        # 检查传入的参数是否合法
        #if action_type not in ['shoulder', 'arm']:
        #    raise ValueError("action_type 必须是 'shoulder' 或 'arm'")
        
        # 【新增】先把当前 episode 的轨迹打包进缓冲区（最多保留最近10个）
        episode_data = {
            "states": list(self.states),
            "actions": list(self.actions),
            "actions_raw": list(self.actions_raw),
            "rewards": list(self.rewards),
            "next_states": list(self.next_states),
            "values": list(self.values),
            "log_probs": list(self.log_probs),
            "dones": list(self.dones),
        }
        self.episode_buffer.append(episode_data)

        # 将缓冲区内所有 episode 的数据拼接成一个大 batch
        all_states, all_actions, all_actions_raw = [], [], []
        all_rewards, all_next_states, all_values = [], [], []
        all_log_probs, all_dones = [], []
        for ep in self.episode_buffer:
            all_states.extend(ep["states"])
            all_actions.extend(ep["actions"])
            all_actions_raw.extend(ep["actions_raw"])
            all_rewards.extend(ep["rewards"])
            all_next_states.extend(ep["next_states"])
            all_values.extend(ep["values"])
            all_log_probs.extend(ep["log_probs"])
            all_dones.extend(ep["dones"])

        # 计算优势函数和回报（基于最近<=10个 episode 的所有样本）
        advantages, returns = self.calculate_advantages(all_rewards, all_values, all_dones)
        if len(advantages) == 0:
            return 0.0

        # 将数据转换为张量
        batch_states = all_states
        batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        batch_actions = torch.tensor(all_actions, dtype=torch.float32).to(device)
        batch_log_probs = torch.tensor(all_log_probs, dtype=torch.float32).to(device)
        # 确保 actions_raw 列表存在且长度匹配
        actions_raw_list = list(all_actions_raw)
        if len(actions_raw_list) != len(all_actions):
            # 如果长度不匹配，从 actions 反推（向后兼容）
            actions_raw_list = [
                torch.atanh(torch.clamp(torch.tensor(a), -0.9999, 0.9999)).item()
                for a in all_actions
            ]
        total_loss = 0
        for _ in range(self.update_epochs):
            # 生成随机索引
            indices = torch.randperm(len(batch_states))

            # 分批处理数据
            for start_idx in range(0, len(batch_states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_x, batch_state, batch_x_graph = [], [], []
                for idx in batch_indices:
                    if idx < len(batch_states):
                        batch_x.append(batch_states[idx][0])
                        batch_state.append(batch_states[idx][1])
                        batch_x_graph.append(batch_states[idx][2])
                        
                if not batch_x:
                    continue

                # --- 【核心修改】这里只计算一次前向传播 ---
                # 不再有 if/else 区分，因为一个 PPO 实例只负责一个智能体
                # dist_batch_values 将是一个列表，包含每个样本的 (distribution, value) 元组
                dist_batch_values = [self.policy(x, s, g) for x, s, g in zip(batch_x, batch_state, batch_x_graph)]
                
                # 提取分布和值
                # dists 是一个 Normal 分布对象的列表
                # values 是一个 value tensor 的列表
                dists = [dv[0] for dv in dist_batch_values]
                values = torch.cat([dv[1].unsqueeze(0) for dv in dist_batch_values]) # 保证values是一个 batch tensor

                # 为整个批次构建一个大的分布对象，方便计算概率
                # 注意:如果你的 network 输出的是一个batch的分布，那就更简单了
                # 但你目前的 network 是对每个样本单独计算的，所以这里需要手动拼接
                # mu_batch = torch.cat([d.mean for d in dists])
                # sigma_batch = torch.cat([d.stddev for d in dists])
                # dist_for_loss = torch.distributions.Normal(mu_batch, sigma_batch)
                
                # 更简单的做法是直接在循环里计算每个样本的loss
                # 这样对于batch size小的情况不会太慢
                
                # 获取当前批次的动作、对数概率、优势、回报等
                # 【修复】确保batch_indices是整数类型，用于索引
                batch_indices_int = batch_indices.cpu().numpy() if isinstance(batch_indices, torch.Tensor) else batch_indices
                batch_actions_curr = batch_actions[batch_indices]
                batch_log_probs_curr = batch_log_probs[batch_indices]
                batch_advantages_curr = batch_advantages[batch_indices]
                batch_returns_curr = batch_returns[batch_indices]
                
                # --- 【核心修改】计算新的对数概率和熵 ---
                # 【关键修复】需要获取原始动作（action_raw）来计算正确的 log_prob
                # 【修复】使用batch_indices_int来索引Python列表
                batch_actions_raw = torch.tensor([actions_raw_list[int(idx)] for idx in batch_indices_int], 
                                                  dtype=torch.float32).to(device)
                
                policy_loss = 0
                entropy = 0
                # 遍历批次中的每一个样本
                for i in range(len(batch_x)):
                    # 【关键修复】使用 action_raw 计算 log_prob，然后应用 tanh squashing 修正
                    # 【修复】简化维度处理：确保 action_raw_i 是正确的形状 [act_dim]
                    action_raw_i = batch_actions_raw[i]
                    if action_raw_i.dim() == 0:  # 标量，需要添加维度
                        action_raw_i = action_raw_i.unsqueeze(0)  # 变成 [1]
                    # 如果已经是1维且长度为act_dim，则保持不变
                    # 注意：对于act_dim=1的情况，shape应该是[1]
                    
                    # 计算原始分布的对数概率
                    new_log_prob_raw = dists[i].log_prob(action_raw_i)
                    
                    # 计算 tanh 后的动作（用于修正项）
                    action_tanh_i = torch.tanh(action_raw_i)
                    # 应用 tanh squashing 修正：log(1 - tanh²(x))
                    tanh_correction = torch.log(1 - action_tanh_i.pow(2) + 1e-6)
                    new_log_prob = (new_log_prob_raw - tanh_correction).sum(dim=-1)
                    
                    ratio = torch.exp(new_log_prob - batch_log_probs_curr[i])
                    
                    surr1 = ratio * batch_advantages_curr[i]
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages_curr[i]
                    
                    policy_loss += -torch.min(surr1, surr2) # 注意这里不带 mean
                    entropy += dists[i].entropy() # 注意这里不带 mean

                policy_loss = policy_loss / len(batch_x) # 最后再取平均
                entropy = entropy / len(batch_x) # 最后再取平均

                # 【新增】对policy_loss进行缩放，控制loss大小（目标：Reward=-90时loss≈7~9）
                # 如果policy_loss太大，缩放它；如果太小，放大它
                policy_loss_scaled = policy_loss * self.policy_loss_scale

                # 值函数损失 (使用Huber loss代替MSE，对异常值更鲁棒)
                # 或者对MSE loss进行裁剪，防止loss过大
                value_loss = nn.MSELoss()(values, batch_returns_curr)
                # 【修复】裁剪value_loss，防止loss过大（限制在100以内）
                value_loss = torch.clamp(value_loss, max=100.0)

                # 总损失 (使用缩放后的policy_loss)
                loss = policy_loss_scaled + self.value_coef * value_loss - self.entropy_coef * entropy

                # 【新增】记录loss分解，帮助诊断问题
                if start_idx == 0:  # 只在第一个batch打印，避免输出过多
                    print(f"  Loss分解: policy_loss(原始)={policy_loss.item():.4f}, policy_loss(缩放后)={policy_loss_scaled.item():.4f} (缩放因子={self.policy_loss_scale})")
                    print(f"  Loss分解: value_loss={value_loss.item():.4f} (权重={self.value_coef}), entropy={entropy.item():.4f} (权重={self.entropy_coef})")
                    print(f"  贡献: policy={policy_loss_scaled.item():.4f}, value={self.value_coef * value_loss.item():.4f}, entropy={-self.entropy_coef * entropy.item():.4f}")
                    print(f"  总loss={loss.item():.4f}")

                # 优化步骤 (保持不变)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()

        # 更新学习率
        self.scheduler.step()

        # 清空轨迹数据
        self.states.clear()
        self.actions.clear()
        self.actions_raw.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("total_loss:", total_loss)
        return total_loss / self.update_epochs