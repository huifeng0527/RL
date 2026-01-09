# ur_control.py

import rtde_receive
import rtde_control
import numpy as np

class URControl:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

    def get_robot_pose(self):
        """ 获取当前机器人姿态 """
        return self.rtde_r.getActualTCPPose()
    
    def move_robot(self, position,t_target):
        """ 控制机器人运动到指定位置 """
        # distance = np.linalg.norm(np.array(position)[:2] - np.array(self.get_robot_pose())[:2])


        # 计算速度（线性）
        speed = 2
        # 可设置一个固定加速度（平滑）
        acceleration =  1
        # print(f"Distance: {distance}, Speed: {speed}, Acceleration: {acceleration}")
        self.rtde_c.moveL(position, speed=speed, acceleration=acceleration,asynchronous=True)

    def disconnect(self):
        """ 断开与 UR 机器人的连接 """
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

