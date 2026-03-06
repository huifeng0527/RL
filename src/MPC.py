import numpy as np

class MPC_Controller:
    def __init__(self, 
                 env_width=15.0, 
                 env_height=10.0, 
                 horizon=10,       # 往前看 10 步 (预测视野)
                 n_samples=1000,   # 每次采样 1000 条轨迹 (越多越准，但越慢)
                 dt=1.0,           # 仿真步长 (通常对应 environment 的 step)
                 v_max=1.0,        # 机器人的物理最大步长 (对应 env.stride_robot)
                 target_dist=5.0,  # ZPD 最佳距离
                 wall_margin=0.5): # 离墙多远开始算撞墙风险
        
        self.W = env_width
        self.H = env_height
        self.H_steps = horizon
        self.N = n_samples
        self.dt = dt
        self.v_max = v_max
        self.d_target = target_dist
        self.wall_margin = wall_margin

    def get_action(self, robot_pos, hand_pos, hand_vel):
        """
        基于采样的 MPC (MPPI 简化版)
        输入: 
            robot_pos: [x, y] 当前机器人位置
            hand_pos:  [x, y] 当前手位置
            hand_vel:  [vx, vy] 当前手速度向量
        输出: 
            best_action: [ax, ay] 归一化的下一步动作 (-1 ~ 1)
        """
        
        # ---------------------------------------------------------
        # 1. 随机采样动作序列 (Sample Random Actions)
        # ---------------------------------------------------------
        # Shape: (N_samples, Horizon, 2)
        # 生成 -1 到 1 之间的随机控制量
        raw_actions = np.random.uniform(-1, 1, size=(self.N, self.H_steps, 2))
        
        # ---------------------------------------------------------
        # 2. 施加物理约束 (Apply Kinematic Constraints)
        # ---------------------------------------------------------
        # 我们假设动作空间是圆形的 (即 dx^2 + dy^2 <= 1)
        # 如果采样点在正方形角上 (模长>1)，需要缩放回圆内，保持方向不变
        magnitudes = np.linalg.norm(raw_actions, axis=2, keepdims=True)
        # 避免除零，虽然概率极小
        magnitudes = np.maximum(magnitudes, 1e-6) 
        
        # 缩放系数: 如果模长>1则缩小，否则保持原样
        scales = np.where(magnitudes > 1, 1.0 / magnitudes, 1.0)
        
        # 归一化的合法动作序列 (-1 ~ 1)
        valid_actions_norm = raw_actions * scales 
        
        # 转换为物理位移 (Physical Displacements)
        # displacements = action * v_max * dt
        displacements = valid_actions_norm * self.v_max * self.dt

        # ---------------------------------------------------------
        # 3. 推演机器人轨迹 (Robot Trajectory Rollout)
        # ---------------------------------------------------------
        # 使用累加函数计算未来每一步的相对位移
        # Shape: (N, H, 2)
        cumulative_disp = np.cumsum(displacements, axis=1)
        
        # 加上当前位置，得到绝对坐标轨迹
        # robot_trajs[i, t, :] 是第 i 条轨迹在时刻 t 的坐标
        robot_trajs = cumulative_disp + robot_pos

        # ---------------------------------------------------------
        # 4. 生成手部预测轨迹 (Predicted Hand Trajectory)
        # ---------------------------------------------------------
        # 假设手在未来 H 步内保持当前速度 (线性预测 Constant Velocity Model)
        # 时间步序列: [1, 2, ..., H] * dt
        time_steps = np.arange(1, self.H_steps + 1) * self.dt
        
        # 预测位移 = v * t
        # shape: (H, 2)
        pred_hand_movements = np.outer(time_steps, hand_vel) 
        
        # future_hand_pos: (H, 2)
        future_hand_pos = hand_pos + pred_hand_movements
        
        # ---------------------------------------------------------
        # 5. 计算代价 (Cost Evaluation)
        # ---------------------------------------------------------
        
        # A. ZPD 代价 (Distance Cost)
        # 利用广播机制: (N, H, 2) - (H, 2) -> numpy 会自动广播 (H, 2) 到每一条采样轨迹上
        diff_vecs = robot_trajs - future_hand_pos
        dists = np.linalg.norm(diff_vecs, axis=2) # Shape: (N, H)
        
        # 我们希望距离接近 target_dist
        # 对未来 H 步的误差平方求和
        zpd_cost = np.sum((dists - self.d_target)**2, axis=1)
        
        # B. 墙壁代价 (Wall Collision Cost)
        # 检查是否出界
        # 左/右边界
        mask_x = (robot_trajs[:, :, 0] < self.wall_margin) | \
                 (robot_trajs[:, :, 0] > self.W - self.wall_margin)
        # 上/下边界
        mask_y = (robot_trajs[:, :, 1] < self.wall_margin) | \
                 (robot_trajs[:, :, 1] > self.H - self.wall_margin)
        
        # 只要轨迹中有任何一步撞墙，就给予巨大惩罚
        # sum(mask) 计算撞墙的步数
        wall_collisions = np.sum(mask_x | mask_y, axis=1)
        wall_cost = wall_collisions * 10000.0 # 软约束转硬约束
        
        # 总代价
        total_cost = zpd_cost + wall_cost
        
        # ---------------------------------------------------------
        # 6. 选择最优动作 (Selection)
        # ---------------------------------------------------------
        best_idx = np.argmin(total_cost)
        
        # 取最优轨迹的第一步动作
        # 注意：这里返回归一化的动作 (-1~1)，因为 gym 环境的 step 会再乘 stride
        best_action_norm = valid_actions_norm[best_idx, 0, :]
        
        return best_action_norm