import random
import numpy as np

def get_hand_move(args:list):

    hand_position = args[0]
    robot_position = args[1]
    stride_hand = args[2]



    # remove by rules
    hand_position = move_by_rule(hand_position, robot_position, stride_hand)

    return hand_position






def move_by_rule(hand_position, robot_position, stride_hand):
    """
    根据规则移动手臂
    """
    if random.random() < 0.1:
        move_hand = np.random.uniform(-1, 1, size=2) * stride_hand  # 10% 概率随机移动
    else:
        dir_vector = robot_position - hand_position
        if np.linalg.norm(dir_vector) > 0:
            dir_vector /= np.linalg.norm(dir_vector)
        move_hand = dir_vector * stride_hand  # Move hand towards robot position

    print(f"move_hand:{move_hand}")

    hand_position += move_hand
    return hand_position


