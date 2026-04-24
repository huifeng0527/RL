import numpy as np

class Advanced_APF:
    def __init__(self, 
                 env_width=15.0, 
                 env_height=10.0,
                 conf=None):
        
        self.w = env_width
        self.h = env_height
        self.center = np.array([env_width/2, env_height/2])
        
        # --- 默认参数配置 (Config) ---
        # 你可以在实例化时覆盖这些参数
        self.p = {
            'z_min': 4,   # ZPD 最佳距离
            'z_max': 6,
            'k_attract': 1.5,     # ZPD 刚度 (拉力大小)
            
            # --- 高级特性 ---
            'elliptical_y': 1,  # 椭圆系数 (1.0=圆, 0.6=扁, 1.5=长)
            'front_bias': 0.0,    # 扇形场权重 (>0 时机器人会被推到手的前方)
            'expand_bias': 0.0,   # 离心场权重 (>0 时机器人倾向于往桌边跑)
            
            # --- 安全特性 ---
            'k_wall': 1,       # 墙壁反弹力 (建议大)
            'wall_margin': 2    # 离墙多远开始反弹
        }
        
        if conf:
            self.p.update(conf)
    def _dist_to_segment(self, point, seg_start, seg_end):
        """
        计算点到线段的最短距离 (核心几何算法)
        point: [x, y]
        seg_start: 手掌中心 (Head)
        seg_end: 手肘位置/手臂末端 (Tail)
        """
        # 向量: 线段起点 -> 终点
        seg_vec = seg_end - seg_start
        # 向量: 线段起点 -> 点
        pt_vec = point - seg_start
        
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq == 0:
            # 线段长度为0，退化为点距离
            return np.linalg.norm(pt_vec)
        
        # 投影系数 t
        t = np.dot(pt_vec, seg_vec) / seg_len_sq
        
        # 限制 t 在 [0, 1] 之间 (线段范围内)
        # t < 0: 最近点是起点 (手掌区域) -> 形成圆形场
        # t > 1: 最近点是终点 (手肘区域) -> 形成圆形场
        # 0 < t < 1: 最近点在线段上 (前臂区域) -> 形成管状场
        t = np.clip(t, 0.0, 1.0)
        
        # 计算线段上的最近点 (Projection Point)
        closest_point = seg_start + t * seg_vec
        
        # 返回距离
        return np.linalg.norm(point - closest_point)
    def _get_potential(self, r_pos, h_pos, h_dir=np.array([0, -1])):
        p = self.p
        
        # --- 1. 定义手臂骨架 (Arm Skeleton) ---
        # 假设手臂长度 (比如 30cm / 10 grid units)
        # 方向是 h_dir 的反方向 (如果手指向前，手臂就在后面)
        # 这里假设 h_dir 指向手指方向，那么手臂在 -h_dir 方向
        arm_len = 0
        
        # 计算手臂末端 (手肘) 坐标
        # 注意：这里用 -h_dir，因为手臂在手掌后方
        # 如果 h_dir 是 [0, -1] (向下)，则 -h_dir 是 [0, 1] (向上/向身前)
        # 请根据你的实际坐标系调整符号！
        # 假设：h_dir 指向机器人要去的方向(前方)，手臂在后方
        arm_end = h_pos - h_dir * arm_len 
        
        # --- 2. 计算"几何距离" (Geometric Distance) ---
        # 机器人距离"手掌+手臂"这个整体的最短距离
        dist_eff = self._dist_to_segment(r_pos, h_pos, arm_end)
        
        # --- 3. 计算 ZPD 势能 ---
        # 现在的 dist_eff 已经是"离身体的距离"了
        # 形状自然变成了胶囊形/钥匙孔形
        if dist_eff<p['z_min']:
            u_zpd = 0.5 * p['k_attract'] * (dist_eff - p['z_min'])**2
        elif dist_eff>p['z_max']:
            u_zpd = 0.5 * p['k_attract'] * (dist_eff - p['z_max'])**2

        else:
            u_zpd = 0
        

        
        # --- 5. 墙壁斥力 (保持不变) ---
        u_wall = 0
        m = p['wall_margin']
        k_w = p['k_wall']
        d_left = r_pos[0]
        d_right = self.w - r_pos[0]
        d_bottom = r_pos[1]
        d_top = self.h - r_pos[1]
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < m:
                u_wall +=  k_w * (m - d)
                
        return u_zpd + u_wall 

    def compute_force(self, robot_pos, hand_pos, hand_direction=None):
        """
        [主接口] 输入坐标，输出力向量 (Action)
        使用数值微分法求梯度: F = - grad(U)
        """
        if hand_direction is None:
            # 如果没给手的方向，自动推断：手指向桌子中心的反方向? 
            # 或者简单默认为向下 [0, -1]
            hand_direction = np.array([0, -1]) 
            
        epsilon = 0.05 # 微分步长
        
        # 计算当前点的势能
        u0 = self._get_potential(robot_pos, hand_pos, hand_direction)
        
        # 计算 X 方向微扰后的势能
        pos_dx = robot_pos + np.array([epsilon, 0])
        ux = self._get_potential(pos_dx, hand_pos, hand_direction)
        
        # 计算 Y 方向微扰后的势能
        pos_dy = robot_pos + np.array([0, epsilon])
        uy = self._get_potential(pos_dy, hand_pos, hand_direction)
        
        # 求导 (梯度下降方向)
        force_x = -(ux - u0) / epsilon
        force_y = -(uy - u0) / epsilon
        
        force = np.array([force_x, force_y])
        
        magnitude = np.linalg.norm(force)
        
        if magnitude < 1e-6:
            return np.zeros(2)
        
        # 1. 归一化方向 (Direction)
        direction = force / magnitude
        
        # 2. 计算速度比例 (Proportional Speed)
        # 使用 tanh 函数来实现平滑饱和
        # sensitivity 控制减速的灵敏度。
        # 值越小，减速越早；值越大，冲得越猛。
        sensitivity = 0.5 
        speed_factor = np.tanh(magnitude * sensitivity)
        
        # 3. 最终输出
        # 输出范围是 [0, 1] 之间的系数，乘以后面的 stride_robot 才是真速度
        # 这样当力很大时，输出接近 1.0 (全速)
        # 当力很小时(比如在 ZPD 中心附近)，输出接近 0.0 (慢速微调)
        final_action = direction * speed_factor
        
        return final_action
        


    # --- 快速切换模式的方法 (给 Task Scheduler 用) ---
    
    def set_mode_standard(self):
        """标准模式: 圆形 ZPD"""
        self.p.update({'elliptical_y': 1.0, 'front_bias': 0.0, 'expand_bias': 0.0})
        
    def set_mode_focused(self):
        """专注模式: 扇形场，防止绕后"""
        self.p.update({'elliptical_y': 1.0, 'front_bias': 2.0, 'expand_bias': 0.0})
        
    def set_mode_expansion(self):
        """扩展模式: 离心场，诱导去边缘"""
        self.p.update({'elliptical_y': 1.0, 'front_bias': 0.5, 'expand_bias': 1.5})