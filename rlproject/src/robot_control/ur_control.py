# ur_control.py

import threading

import rtde_receive
import rtde_control
import numpy as np

class URControl:
    def __init__(self, robot_ip, control_frequency=None):
        self.robot_ip = robot_ip
        self.control_lock = threading.RLock()
        self.control_frequency_requested = (
            None if control_frequency is None else float(control_frequency)
        )
        self.control_frequency_configured = False
        # 建立控制和接收连接
        if self.control_frequency_requested is None:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        else:
            try:
                self.rtde_c = rtde_control.RTDEControlInterface(
                    self.robot_ip,
                    frequency=self.control_frequency_requested,
                )
                self.control_frequency_configured = True
            except TypeError:
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
        with self.control_lock:
            self.rtde_c.moveL(
                target_pose,
                speed,
                acceleration,
                asynchronous=False,
            )

    # ---------------------------------------------------------
    # 方案一：位置伺服控制 (强烈推荐用于强化学习高频控制)
    # ---------------------------------------------------------
    def servo_robot(
        self,
        target_pose,
        dt,
        lookahead_time=0.1,
        gain=600,
    ):
        """控制机器人平滑滑动到目标位置。"""
        velocity = 0.0
        acceleration = 0.0
        with self.control_lock:
            return self.rtde_c.servoL(
                target_pose,
                velocity,
                acceleration,
                dt,
                lookahead_time,
                gain,
            )

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
        with self.control_lock:
            self.rtde_c.speedL(velocity_vector, acceleration, time)

    def servo_stop(self):
        """停止位置伺服。"""
        with self.control_lock:
            self.rtde_c.servoStop()

    def stop_robot(self):
        """ 紧急刹车/停止伺服 """
        with self.control_lock:
            self.rtde_c.servoStop()
            self.rtde_c.speedStop()

    def disconnect(self, stop=True):
        """ 断开与 UR 机器人的连接 """
        with self.control_lock:
            if stop:
                self.rtde_c.servoStop()
                self.rtde_c.speedStop()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()