from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SpatioTemporalAttention(nn.Module):
    """
    来自 tsattenGrasp 的时空注意力融合模块：
    - 用注意力权重融合 CNN 特征与状态特征
    - 这里保留实现，方便在 PPO 中随时启用
    """
    def __init__(self, x_dim, state_dim, hidden_dim=200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(x_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim + x_dim, hidden_dim)

    def forward(self, x, state):
        # x/state 均为 [feature_dim]，扩展 batch 维度以复用 tsattenGrasp 逻辑
        q = self.query(x.unsqueeze(0))
        k = self.key(state.unsqueeze(0))
        v = self.value(state.unsqueeze(0))

        scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=x.device))
        scores = torch.matmul(q, k.transpose(0, 1)) / scale
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, v)

        combined = torch.cat([x, attended_values.squeeze(0)], dim=-1)
        return self.proj(combined)


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

        # tsattenGrasp 时空注意力融合模块（调用在 forward 中默认注释）
        self.attention_fusion = SpatioTemporalAttention(x_dim=100, state_dim=100, hidden_dim=200)
        
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
        # 【修复】提高初始探索噪声：从-1.0（对应sigma≈0.37）改为-0.5（对应sigma≈0.61）
        # 问题：之前sigma太小导致探索不足，网络过早收敛到次优策略
        # 解决方案：增加初始探索率，让网络有更多机会探索好的策略
        self.actor_log_sigma = nn.Parameter(torch.tensor([-0.5]))  # 初始sigma ≈ 0.61，增加探索
        
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

        # 使用 LMF 融合图像特征和状态特征（默认启用）
        img_feat = normalized_data1.unsqueeze(0)   # [1, 100]
        state_feat = normalized_data2.unsqueeze(0) # [1, 100]
        fused_feat = self.lmf(img_feat, state_feat).squeeze(0)  # [200]

        # === tsattenGrasp 时空注意力融合（功能已合入，默认注释掉） ===
        # 启用方法：注释掉上面的 LMF fused_feat 行，并取消下方一行的注释
        # fused_feat = self.attention_fusion(normalized_data1, normalized_data2)

        # === 无 LMF 版本：直接拼接图像和状态特征（备用实现） ===
        # fused_feat = torch.cat((normalized_data1, normalized_data2), dim=-1)

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
        # 【修复】提高policy_loss缩放因子，增强学习信号
        # 问题：0.1太小，导致学习信号太弱，网络学不到好的策略
        # 解决方案：提高到0.5，让网络对reward变化更敏感
        self.policy_loss_scale = 0.5  # 从0.1提高到0.5，增强学习信号
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

        # 学习率和优化器参数
        self.lr = 3e-4  # 提高学习率
        self.lr_decay = 0.999  # 减缓学习率衰减

        # PPO更新参数
        self.update_epochs = 8  # 【优化】增加更新次数，提高学习稳定性（从4增加到8）
        self.batch_size = 64  # 批大小

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
        
        # 【新增】自适应探索率调整：记录学习历史
        self.learning_history = {
            'losses': deque(maxlen=20),           # 最近20个episode的总loss
            'policy_losses': deque(maxlen=20),    # 最近20个episode的policy_loss
            'rewards': deque(maxlen=20),          # 最近20个episode的累计reward
            'reward_sums': deque(maxlen=20),      # 最近20个episode的reward总和
        }
        # 探索率调整参数
        # 【修复】调整探索率范围，保持足够的探索能力
        self.sigma_adjustment_rate = 0.02  # 每次调整的幅度（2%，更保守）
        self.sigma_min = 0.2   # 【修复】提高最小sigma从0.15到0.2，避免过早收敛（log_sigma ≈ -1.61）
        self.sigma_max = 0.8   # 【修复】提高最大sigma从0.5到0.8，允许更多探索（log_sigma ≈ -0.22）
        self.min_episodes_before_adjust = 20  # 至少20个episode后才开始调整
        self.adjust_interval = 3  # 每3个episode调整一次，避免过于频繁
        self.last_adjust_episode = 0  # 记录上次调整的episode
    
    def choose_action(self, episode_num, obs, x_graph, explore=None):
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph

        # 【修复】确保所有输入都移到正确的设备
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
        
        # 【修复】确保 state 也在正确的设备上
        if isinstance(state, torch.Tensor):
            state = state.to(device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32).to(device)

        # 【修复】加快epsilon衰减，让网络更快从随机探索转向策略利用
        # 原来：max(0.1, 0.90 - episode_num * 0.0001) - 衰减太慢，1000个episode后还是0.8
        # 现在：max(0.05, 0.90 - episode_num * 0.001) - 1000个episode后降到0.05
        epsilon = max(0.05, 0.90 - episode_num * 0.001)
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
        # 【修复】降低缩放阈值，更早进行缩放，避免优势函数过大导致梯度爆炸
        # 问题：如果advantages太大，可能导致梯度不稳定，网络学习失败
        # 解决方案：降低阈值到100.0，更早进行缩放，保持学习稳定性
        advantages_std = advantages.std()
        if advantages_std > 100.0:  # 【修复】降低阈值从200.0到100.0，更早缩放保持稳定
            # 缩放因子：将标准差压缩到100以内，但保留相对大小
            scale_factor = 100.0 / (advantages_std + 1e-8)
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
    
    def _adjust_exploration_rate(self, episode_num):
        """
        根据学习情况自适应调整探索率（sigma）
        改进策略：
        1. 更保守的调整：减小调整幅度，增加调整间隔
        2. 更严格的降低条件：需要return为正且稳定上升
        3. 更积极的增加条件：如果return为负或波动大，立即增加探索率
        4. 训练阶段保护：早期保持高探索率
        """
        # 检查是否满足调整条件
        if len(self.learning_history['losses']) < 10:  # 需要至少10个episode的数据
            return
        
        # 检查调整间隔
        if episode_num - self.last_adjust_episode < self.adjust_interval:
            return
        
        # 早期训练阶段保护：前N个episode保持高探索率
        if episode_num < self.min_episodes_before_adjust:
            return
        
        losses = list(self.learning_history['losses'])
        policy_losses = list(self.learning_history['policy_losses'])
        rewards = list(self.learning_history['reward_sums'])
        
        # 计算最近10个episode的趋势（使用更多数据更稳定）
        recent_losses = losses[-10:]
        recent_policy_losses = policy_losses[-10:]
        recent_rewards = rewards[-10:]
        
        # 1. 分析loss趋势
        loss_trend = (recent_losses[-1] - recent_losses[0]) / max(abs(recent_losses[0]), 1e-6)
        # 2. 分析reward趋势和绝对值
        reward_mean = sum(recent_rewards) / len(recent_rewards)
        reward_std = (sum((x - reward_mean)**2 for x in recent_rewards) / len(recent_rewards))**0.5
        reward_trend = (recent_rewards[-1] - recent_rewards[0]) / max(abs(recent_rewards[0]), 1e-6) if recent_rewards[0] != 0 else 0
        # 3. 分析policy_loss的平均值和稳定性
        policy_loss_mean = sum(recent_policy_losses) / len(recent_policy_losses)
        policy_loss_std = (sum((x - policy_loss_mean)**2 for x in recent_policy_losses) / len(recent_policy_losses))**0.5
        
        # 获取当前的sigma
        current_log_sigma = self.policy.actor_log_sigma.item()
        current_sigma = torch.exp(self.policy.actor_log_sigma).item()
        
        # 决策：是否应该调整探索率
        should_decrease = False  # 是否应该降低探索率
        should_increase = False  # 是否应该增加探索率
        reason = ""
        
        # 【优先】情况1：如果return为负或波动很大，增加探索率
        if reward_mean < 0 or reward_std > 100:
            should_increase = True
            reason = f"return为负({reward_mean:.2f})或波动大(std={reward_std:.2f})，需要更多探索"
        
        # 情况2：如果return持续为正且稳定上升，可以考虑降低探索率
        elif reward_mean > 50 and reward_trend > 0.1 and reward_std < 50:
            # 进一步检查：loss是否也在下降
            if loss_trend < -0.05:
                should_decrease = True
                reason = f"return为正且稳定上升(mean={reward_mean:.2f}, trend={reward_trend:.2%})，loss下降"
        
        # 情况3：loss持续下降且reward上升（更严格的条件）
        elif loss_trend < -0.15 and reward_trend > 0.15 and reward_mean > 0:
            should_decrease = True
            reason = f"loss大幅下降({loss_trend:.2%})且reward大幅上升({reward_trend:.2%})"
        
        # 情况4：loss上升或reward下降，增加探索率
        elif loss_trend > 0.15 or (reward_trend < -0.15 and reward_mean < 0):
            should_increase = True
            reason = f"loss上升({loss_trend:.2%})或reward下降({reward_trend:.2%})"
        
        # 情况5：policy_loss很小且稳定，且return为正（更严格的条件）
        elif policy_loss_mean < 0.3 and policy_loss_std < 0.15 and reward_mean > 30:
            should_decrease = True
            reason = f"policy_loss很小且稳定(mean={policy_loss_mean:.3f}, std={policy_loss_std:.3f})，return为正"
        
        # 情况6：policy_loss很大且不稳定，需要更多探索
        elif policy_loss_mean > 2.5 and policy_loss_std > 1.2:
            should_increase = True
            reason = f"policy_loss大且不稳定(mean={policy_loss_mean:.3f}, std={policy_loss_std:.3f})"
        
        # 执行调整
        if should_decrease and current_sigma > self.sigma_min:
            # 降低探索率：减小sigma（减小log_sigma）
            new_log_sigma = current_log_sigma - self.sigma_adjustment_rate
            new_sigma = torch.exp(torch.tensor(new_log_sigma)).item()
            if new_sigma >= self.sigma_min:
                # 【修复】确保新 tensor 在正确的设备上
                self.policy.actor_log_sigma.data = torch.tensor([new_log_sigma], device=self.policy.actor_log_sigma.device)
                self.last_adjust_episode = episode_num
                print(f"  【自适应探索率】降低探索率: {reason}, sigma: {current_sigma:.4f} -> {new_sigma:.4f}")
        
        elif should_increase and current_sigma < self.sigma_max:
            # 增加探索率：增大sigma（增大log_sigma）
            new_log_sigma = current_log_sigma + self.sigma_adjustment_rate
            new_sigma = torch.exp(torch.tensor(new_log_sigma)).item()
            if new_sigma <= self.sigma_max:
                # 【修复】确保新 tensor 在正确的设备上
                self.policy.actor_log_sigma.data = torch.tensor([new_log_sigma], device=self.policy.actor_log_sigma.device)
                self.last_adjust_episode = episode_num
                print(f"  【自适应探索率】增加探索率: {reason}, sigma: {current_sigma:.4f} -> {new_sigma:.4f}")
    
    def learn(self):
        """
        学习函数：每次调用时学习当前累积的所有episode数据
        - 调用端已控制每10个episode才调用一次
        - 直接使用 self.states 等列表中已累积的数据进行学习
        """
        # 检查是否有数据
        if len(self.rewards) == 0:
            print("  警告：没有数据可学习，返回0")
            return 0.0
        
        print(f"  【开始学习】使用累积的 {len(self.rewards)} 个样本进行学习")
        
        # 直接使用已累积的数据（已包含10个episode的数据）
        all_states = self.states
        all_actions = self.actions
        all_actions_raw = self.actions_raw
        all_rewards = self.rewards
        all_next_states = self.next_states
        all_values = self.values
        all_log_probs = self.log_probs
        all_dones = self.dones

        # 计算优势函数和回报
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
        total_policy_loss = 0  # 【新增】累计所有batch的policy_loss
        batch_count = 0  # 【新增】记录batch数量
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
                
                # 【新增】累计policy_loss（用于自适应调整）
                total_policy_loss += policy_loss.item()
                batch_count += 1

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
        
        # 【新增】记录当前episode的学习指标，用于自适应探索率调整
        avg_loss = total_loss / self.update_epochs
        reward_sum = sum(all_rewards) if all_rewards else 0.0
        policy_loss_avg = total_policy_loss / batch_count if batch_count > 0 else 0.0
        
        self.learning_history['losses'].append(avg_loss)
        self.learning_history['policy_losses'].append(policy_loss_avg)
        self.learning_history['rewards'].append(reward_sum)
        self.learning_history['reward_sums'].append(reward_sum)
        
        # 【新增】根据学习情况自适应调整探索率
        # 使用learning_history的长度作为episode编号（因为每个episode都会添加一次数据）
        episode_num = len(self.learning_history['losses'])
        self._adjust_exploration_rate(episode_num)

        # 学习完成，清空轨迹数据，准备下一轮累积
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
        print(f"  【学习完成】已清空数据，等待下一轮10个episode...")
        print("total_loss:", total_loss)
        return total_loss / self.update_epochs