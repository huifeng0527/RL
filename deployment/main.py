"""
部署主程序
用于实际机器人控制和实时推理
"""
import traceback
import numpy as np
import cv2
import time
import random
from collections import deque
from matplotlib import pyplot as plt

from stable_baselines3 import SAC

# 导入自定义模块
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deployment.robot_control.ur_control import URControl
from deployment.vision.camera_calibration import CameraCalibration
from deployment.vision.hand_detection import HandDetection
from deployment.vision.hand_movement import get_hand_move
from deployment.vision.workspace import get_workspace
from deployment.vision.camerastream import CameraStream
from deployment.utils.config_loader import ConfigLoader
from deployment.utils.renderer import DeploymentRenderer
from common.observation_space import build_observation

from ultralytics import YOLO


class DeploymentController:
    """部署控制器类"""
    
    def __init__(self, config_dir=None):
        """
        初始化部署控制器
        
        Args:
            config_dir: 配置文件目录
        """
        # 加载配置
        self.config_loader = ConfigLoader(config_dir)
        self.robot_config = self.config_loader.get_robot_config()
        self.camera_config = self.config_loader.get_camera_config()
        self.model_config = self.config_loader.get_model_config()
        self.hand_config = self.config_loader.get_hand_detection_config()
        
        # 初始化组件
        self._init_robot()
        self._init_camera()
        self._init_vision()
        self._init_model()
        self._init_visualization()
        
        # 状态变量
        self.trajectory_robot = deque(maxlen=60)
        self.trajectory = deque(maxlen=60)
        self.distance_list = deque(maxlen=1000)
        self.last_action = np.zeros(2)
        self.last_hand = np.zeros(2)
        self.terminated = False
        self.terminated_step = 0
        self.step_count = 0
        self.fixed_point = [10, 10]
        
        # 环境参数
        self.w_env = self.model_config['env_width']
        self.h_env = self.model_config['env_height']
        self.stride_robot = self.model_config['stride_robot']
        
    def _init_robot(self):
        """初始化机器人控制"""
        robot_ip = self.robot_config['robot_ip']
        self.robot_control = URControl(robot_ip)
        print(f"机器人控制已初始化，IP: {robot_ip}")
    
    def _init_camera(self):
        """初始化摄像头"""
        camera_id = self.camera_config['camera_id']
        width = self.camera_config['desired_width']
        height = self.camera_config['desired_height']
        
        self.cap = CameraStream(width, height, camera_id)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        print(f"摄像头已初始化，分辨率: {width}x{height}")
    
    def _init_vision(self):
        """初始化视觉处理"""
        # 相机标定
        calib_path = self.config_loader.resolve_path(
            self.camera_config['calibration_matrix_path']
        )
        homography_path = self.config_loader.resolve_path(
            self.camera_config['homography_matrix_path']
        )
        self.cali = CameraCalibration(calib_path, homography_path)
        
        # 手部检测
        self.hand_detector = HandDetection(
            max_num_hands=self.hand_config['max_num_hands'],
            min_detection_confidence=self.hand_config['min_detection_confidence'],
            min_tracking_confidence=self.hand_config['min_tracking_confidence']
        )
        
        # YOLO模型
        yolo_path = self.config_loader.resolve_path(
            self.model_config['yolo_model_path']
        )
        self.cv_model = YOLO(yolo_path)
        print("视觉处理模块已初始化")
    
    def _init_model(self):
        """初始化RL模型"""
        model_path = self.config_loader.resolve_path(
            self.model_config['model_path']
        )
        
        # 创建临时环境用于加载模型（需要观察空间和动作空间）
        from simulation.environments import CustomEnv
        temp_env = CustomEnv()
        temp_env.random = False
        
        self.model = SAC.load(
            model_path,
            env=temp_env,
            custom_objects={
                "observation_space": temp_env.observation_space,
                "action_space": temp_env.action_space
            }
        )
        print(f"模型已加载: {model_path}")
    
    def _init_visualization(self):
        """初始化可视化"""
        if self.model_config.get('enable_visualization', True):
            self.renderer = DeploymentRenderer(
                grid_size=10,
                cell_size=50
            )
        
        if self.model_config.get('enable_histogram', True):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(5, 4))
            self.bins = np.linspace(0, 10, 15)
            self.ax.set_title("Distance Distribution (实时更新)")
            self.ax.set_xlabel("Distance Value")
            self.ax.set_ylabel("Frequency")
            plt.show(block=False)
    
    def process_frame(self, frame):
        """
        处理一帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            tuple: (处理后的图像, 机器人像素位置, 手部像素位置)
        """
        # 畸变矫正
        undistorted_frame = self.cali.undistort_frame(frame)
        
        # 获取工作空间
        undistorted_frame = get_workspace(undistorted_frame)
        
        # 旋转图像
        undistorted_frame = cv2.rotate(
            undistorted_frame, 
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
        
        h, w = undistorted_frame.shape[:2]
        
        # YOLO检测机器人
        results = self.cv_model.predict(
            undistorted_frame, 
            conf=self.model_config['yolo_confidence'],
            save=False,
            imgsz=self.model_config['yolo_image_size'],
            verbose=False
        )
        
        robot_pixel = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2 - x1 > 100:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                robot_pixel = (cx, cy)
                self.trajectory_robot.append([cx, cy])
                cv2.rectangle(
                    undistorted_frame, 
                    (x1, y1), 
                    (x2, y2), 
                    (255, 0, 255), 
                    6
                )
        
        # 绘制机器人轨迹
        if len(self.trajectory_robot) >= 2:
            i = 2
            for j in range(1, len(self.trajectory_robot)):
                cv2.line(
                    undistorted_frame,
                    self.trajectory_robot[j - 1],
                    self.trajectory_robot[j],
                    (0, 255, 255),
                    int(i // 2)
                )
                i += 0.2
        
        # 手部检测
        undistorted_frame, hand_positions = self.hand_detector.process_frame(
            undistorted_frame
        )
        
        hand_pixel = None
        if hand_positions:
            hand_pixel = hand_positions[0]
        
        return undistorted_frame, robot_pixel, hand_pixel, h, w
    
    def compute_observation(self, robot_pixel, hand_pixel, h, w):
        """
        计算观察值
        
        Args:
            robot_pixel: 机器人像素位置
            hand_pixel: 手部像素位置
            h: 图像高度
            w: 图像宽度
            
        Returns:
            np.ndarray: 观察值
        """
        if robot_pixel is None or hand_pixel is None:
            return None
        
        # 转换为环境坐标
        position_robot_env = (
            robot_pixel[0] * self.w_env / w,
            robot_pixel[1] * self.h_env / h
        )
        
        if not self.terminated:
            position_hand_env = get_hand_move([
                np.array(self.last_hand, dtype=np.float32),
                np.array(position_robot_env, dtype=np.float32),
                self.stride_robot * 0.5
            ])
        else:
            self.terminated_step += 1
            if self.terminated_step > self.model_config.get('termination_steps', 4):
                self.fixed_point = [
                    random.randint(0, self.w_env),
                    self.h_env
                ]
                self.terminated = False
                self.terminated_step = 0
            position_hand_env = self.last_hand
        
        # 计算距离和边界
        robot = np.array([position_robot_env], dtype=np.float32)
        hand = np.array([position_hand_env], dtype=np.float32)
        stride_hand = np.linalg.norm(hand - self.last_hand)
        distance_to_object = np.linalg.norm(robot - hand)
        boundary = np.array([
            min(
                position_robot_env[0],
                position_robot_env[1],
                self.w_env - position_robot_env[0],
                self.h_env - position_robot_env[1]
            )
        ], dtype=np.float32)
        
        # 计算到手臂的距离
        from simulation.environments import CustomEnv
        temp_env = CustomEnv()
        dist_arm = temp_env.dist_point_to_segment_correct(
            robot.flatten(),
            hand.flatten(),
            [self.w_env, self.h_env]
        )[0]
        
        # 构建观察值
        obs = build_observation(
            robot_position=robot.flatten(),
            hand_position=hand.flatten(),
            last_action=self.last_action.flatten(),
            current_distance=distance_to_object,
            boundary=boundary[0],
            dist_arm=dist_arm,
            fixed_point=self.fixed_point,
            stride_robot=self.stride_robot,
            stride_hand=stride_hand,
            env_size=[self.w_env, self.h_env]
        )
        
        self.last_hand = hand
        return obs, position_robot_env, position_hand_env
    
    def execute_action(self, action, robot_pixel, h, w):
        """
        执行动作
        
        Args:
            action: 动作数组
            robot_pixel: 机器人当前像素位置
            h: 图像高度
            w: 图像宽度
        """
        if robot_pixel is None:
            return
        
        # 转换为像素坐标的动作
        action_pixel = action * np.array([w / self.w_env, h / self.h_env]) * self.stride_robot
        
        # 更新机器人位置
        new_robot_pixel = np.array(robot_pixel) + action_pixel
        
        # 转换为世界坐标
        position_robot_world = self.cali.pixel_to_world(new_robot_pixel)
        
        # 控制机器人移动
        rx, ry, rz = self.robot_config['robot_orientation']
        z_height = self.robot_config['robot_z_height']
        
        self.robot_control.move_robot(
            [position_robot_world[0], position_robot_world[1], z_height, rx, ry, rz],
            1 / self.model_config['control_frequency']
        )
    
    def update_visualization(self, obs, position_robot_env, position_hand_env):
        """更新可视化"""
        if obs is None:
            return
        
        # 更新渲染器
        if hasattr(self, 'renderer'):
            self.renderer.render(
                obs[:2],
                obs[2:4],
                self.fixed_point,
                self.trajectory,
                self.fixed_point
            )
        
        # 更新直方图
        if hasattr(self, 'fig') and obs[8] < 2:
            self.terminated = True
        
        if hasattr(self, 'fig'):
            dist_arm = obs[8]
            self.distance_list.append(float(dist_arm))
            self.ax.clear()
            self.ax.hist(
                list(self.distance_list),
                bins=self.bins,
                color='skyblue',
                alpha=0.7,
                density=True
            )
            self.ax.set_title("Distance Distribution")
            self.ax.set_xlabel("density")
            self.ax.set_ylabel("Dis")
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            plt.draw()
            plt.pause(0.001)
    
    def run(self):
        """运行主循环"""
        last_trigger_time = time.time()
        freq = self.model_config['control_frequency']
        max_steps = self.model_config.get('max_steps', 1000)
        
        try:
            while self.step_count < max_steps:
                ret, frame = self.cap.read()
                if not ret:
                    print("错误: 无法从摄像头读取帧")
                    break
                
                # 处理帧
                processed_frame, robot_pixel, hand_pixel, h, w = self.process_frame(frame)
                
                # 显示图像
                if robot_pixel:
                    cv2.circle(
                        processed_frame,
                        robot_pixel,
                        10,
                        (0, 255, 0),
                        -1
                    )
                if hand_pixel:
                    cv2.circle(
                        processed_frame,
                        hand_pixel,
                        10,
                        (255, 0, 0),
                        -1
                    )
                
                cv2.putText(
                    processed_frame,
                    f"Step: {self.step_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.imshow('Frame', processed_frame)
                
                # 控制循环
                now = time.time()
                if now - last_trigger_time > 1 / freq:
                    if robot_pixel and hand_pixel:
                        # 计算观察值
                        result = self.compute_observation(robot_pixel, hand_pixel, h, w)
                        if result is None:
                            continue
                        
                        obs, position_robot_env, position_hand_env = result
                        
                        if not self.terminated:
                            # 预测动作
                            action, _states = self.model.predict(obs, deterministic=True)
                            self.last_action = action
                            
                            print(f"obs: {obs}\n action: {action}")
                            
                            # 执行动作
                            self.execute_action(action, robot_pixel, h, w)
                            
                            self.step_count += 1
                            self.trajectory.append(position_robot_env)
                            
                            # 更新可视化
                            self.update_visualization(
                                obs,
                                position_robot_env,
                                position_hand_env
                            )
                    
                    last_trigger_time = now
                
                # 检查退出
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        
        except Exception as e:
            print(f"错误发生: {e}")
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'robot_control'):
            self.robot_control.disconnect()
        if hasattr(self, 'hand_detector'):
            self.hand_detector.release()
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'renderer'):
            self.renderer.quit()
        if hasattr(self, 'fig'):
            plt.ioff()
            plt.close()


def main():
    """主函数"""
    controller = DeploymentController()
    controller.run()


if __name__ == "__main__":
    main()

