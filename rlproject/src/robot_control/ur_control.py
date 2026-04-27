# ur_control.py

import rtde_receive
import rtde_control
import numpy as np

class URControl:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        # 建立控制和接收连接
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

    def get_robot_pose(self):
        """ 获取当前机器人末端 TCP 姿态 [x, y, z, rx, ry, rz] """
        return self.rtde_r.getActualTCPPose()
    
    # ---------------------------------------------------------
    # 方案零：Movel 点对点移动 (用于跳跃到新目标点)
    # ---------------------------------------------------------
    def move_robot(self, target_pose, speed=0.2, acceleration=0.2):
        """
        使用 movel 指令移动到指定位置（点对点移动）。
        target_pose: [x, y, z, rx, ry, rz] (单位: m, rad)
        speed: 工具速度 (m/s)
        acceleration: 工具加速度 (m/s^2)
        """
        self.rtde_c.moveL(target_pose, speed, acceleration, asynchronous=False)

    # ---------------------------------------------------------
    # 方案一：位置伺服控制 (强烈推荐用于强化学习高频控制)
    # ---------------------------------------------------------
    def servo_robot(self, target_pose, dt):
        """ 
        控制机器人平滑滑动到指定位置，无顿挫。
        target_pose:[x, y, z, rx, ry, rz] (单位: m, rad)
        dt: 控制周期的步长 (例如 1/freq)
        """
        # 在 servoL 中，velocity 和 acceleration 设为 0，由底层控制器接管
        velocity = 0.0
        acceleration = 0.0
        
        # --- 核心平滑参数 ---
        # lookahead_time (前瞻时间): 范围[0.03, 0.2]。
        # 值越大轨迹越平滑，但跟随滞后越大；值越小跟随越紧，但容易抖动。默认 0.1 较好。
        lookahead_time = 0.1
        
        # gain (比例增益): 范围 [100, 2000]。
        # 值越大响应越快，目标追踪越紧密。
        gain =400
        
        # 调用 servoL
        self.rtde_c.servoL(target_pose, velocity, acceleration, dt, lookahead_time, gain)

    # ---------------------------------------------------------
    # 方案二：速度控制 (如果你想直接输出速度)
    # ---------------------------------------------------------
    def speed_robot(self, velocity_vector):
        """
        直接给机械臂下发速度指令。
        velocity_vector:[vx, vy, vz, vrx, vry, vrz] (单位: m/s, rad/s)
        """
        # 这里的加速度决定了速度突变时的柔顺度。设小一点动作会非常丝滑
        acceleration = 0.5 
        time = 0.0 # 0.0 表示一直按这个速度走，直到下一个 speedL 指令覆盖
        self.rtde_c.speedL(velocity_vector, acceleration, time)

    def stop_robot(self):
        """ 紧急刹车/停止伺服 """
        self.rtde_c.servoStop()
        self.rtde_c.speedStop()

    def disconnect(self):
        """ 断开与 UR 机器人的连接 """
        self.stop_robot() # 断开前先停止运动
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()