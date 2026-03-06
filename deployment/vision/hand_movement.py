"""
手部移动模块
用于计算手部的移动
"""
import random
import numpy as np


def get_hand_move(args):
    """
    根据规则移动手部
    
    Args:
        args: [hand_position, robot_position, stride_hand]
            - hand_position: 当前手部位置
            - robot_position: 机器人位置
            - stride_hand: 手部步长
            
    Returns:
        np.ndarray: 新的手部位置
    """
    hand_position = args[0]
    robot_position = args[1]
    stride_hand = args[2]

    # 根据规则移动
    hand_position = move_by_rule(hand_position, robot_position, stride_hand)

    return hand_position


def move_by_rule(hand_position, robot_position, stride_hand):
    """
    根据规则移动手臂
    
    Args:
        hand_position: 当前手部位置
        robot_position: 机器人位置
        stride_hand: 手部步长
        
    Returns:
        np.ndarray: 新的手部位置
    """
    if random.random() < 0.1:
        # 10% 概率随机移动
        move_hand = np.random.uniform(-1, 1, size=2) * stride_hand
    else:
        # 90% 概率向机器人移动
        dir_vector = robot_position - hand_position
        if np.linalg.norm(dir_vector) > 0:
            dir_vector /= np.linalg.norm(dir_vector)
        move_hand = dir_vector * stride_hand

    hand_position += move_hand
    return hand_position

