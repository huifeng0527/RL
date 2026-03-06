"""
UR机器人控制模块
"""
import rtde_receive
import rtde_control
import numpy as np


class URControl:
    """UR机器人控制类"""
    
    def __init__(self, robot_ip):
        """
        初始化机器人控制
        
        Args:
            robot_ip: 机器人IP地址
        """
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

    def get_robot_pose(self):
        """
        获取当前机器人姿态
        
        Returns:
            list: [x, y, z, rx, ry, rz] 机器人位姿
        """
        return self.rtde_r.getActualTCPPose()
    
    def move_robot(self, position, t_target):
        """
        控制机器人运动到指定位置
        
        Args:
            position: [x, y, z, rx, ry, rz] 目标位置和姿态
            t_target: 目标时间（未使用，保留兼容性）
        """
        # 计算速度（线性）
        speed = 2
        # 可设置一个固定加速度（平滑）
        acceleration = 1
        self.rtde_c.moveL(position, speed=speed, acceleration=acceleration, asynchronous=True)

    def disconnect(self):
        """断开与UR机器人的连接"""
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

