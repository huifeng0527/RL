"""
共享的观察空间定义
确保仿真和部署使用相同的观察空间结构
"""
import numpy as np
from gymnasium.spaces import Box


def get_observation_space(env_width, env_height, grid_size, stride_robot_max, stride_hand_max):
    """
    获取观察空间定义
    
    Args:
        env_width: 环境宽度
        env_height: 环境高度
        grid_size: 网格大小
        stride_robot_max: 机器人最大步长
        stride_hand_max: 手部最大步长
        
    Returns:
        observation_space: Gymnasium观察空间
    """
    observation_shape = 17  # 总共17维
    
    observation_space = Box(
        low=0, 
        high=np.array([
            env_width, env_height,           # robot position (2)
            env_width, env_height,            # hand position (2)
            1, 1,                            # last action (2)
            (2**0.5) * grid_size,            # current distance (1)
            0.5 * grid_size,                 # boundary (1)
            0.5 * grid_size,                 # dist_arm (1)
            2 * grid_size, grid_size,        # fixed_point (2)
            stride_robot_max,                # stride_robot (1)
            stride_hand_max,                 # stride_hand (1)
            env_width, env_height            # env_size (2)
        ]), 
        shape=(observation_shape,), 
        dtype=np.float32
    )
    
    return observation_space


def parse_observation(obs):
    """
    解析观察值（用于调试和可视化）
    
    Args:
        obs: 观察值数组
        
    Returns:
        dict: 解析后的观察值字典
    """
    return {
        'robot_position': obs[0:2],
        'hand_position': obs[2:4],
        'last_action': obs[4:6],
        'current_distance': obs[6],
        'boundary': obs[7],
        'dist_arm': obs[8],
        'fixed_point': obs[9:11],
        'stride_robot': obs[11],
        'stride_hand': obs[12],
        'env_size': obs[13:15]
    }


def build_observation(robot_position, hand_position, last_action, 
                     current_distance, boundary, dist_arm, 
                     fixed_point, stride_robot, stride_hand, env_size):
    """
    构建观察值数组
    
    Args:
        robot_position: 机器人位置 [x, y]
        hand_position: 手部位置 [x, y]
        last_action: 上一步动作 [dx, dy]
        current_distance: 当前距离
        boundary: 到边界的最近距离
        dist_arm: 到手臂的距离
        fixed_point: 固定点位置 [x, y]
        stride_robot: 机器人步长
        stride_hand: 手部步长
        env_size: 环境尺寸 [width, height]
        
    Returns:
        np.array: 观察值数组
    """
    return np.concatenate((
        [np.array(robot_position)],
        [np.array(hand_position)],
        [np.array(last_action)],
        [np.array([current_distance])],
        [np.array([boundary])],
        [np.array([dist_arm])],
        [np.array(fixed_point)],
        [np.array([stride_robot])],
        [np.array([stride_hand])],
        [np.array(env_size)]
    ))

