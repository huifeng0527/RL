"""
部署渲染器模块
用于可视化部署过程中的状态
"""
import pygame
import os


class DeploymentRenderer:
    """部署渲染器类"""
    
    def __init__(self, grid_size=10, cell_size=50, window_size=None, hand_image_path=None):
        """
        初始化渲染器
        
        Args:
            grid_size: 网格大小
            cell_size: 每个网格单元的像素大小
            window_size: 窗口大小
            hand_image_path: 手部图像路径
        """
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = window_size or (grid_size * cell_size * 1.5, grid_size * cell_size)
        self.window = pygame.display.set_mode(self.window_size)
        self.canvas = pygame.Surface(self.window_size)
        self.canvas.fill((255, 255, 255))

        # 预加载手部图像
        self.hand_image_path = hand_image_path
        if hand_image_path and os.path.exists(hand_image_path):
            self.virus_image = pygame.image.load(hand_image_path).convert_alpha()
        else:
            self.virus_image = pygame.Surface((self.cell_size * 2, self.cell_size * 2))
            self.virus_image.fill((0, 255, 0))
        
        self.font = pygame.font.Font(None, 24)

    def render(self, robot_position, hand_position, fixed_point, trajectory_points, blocking_point=None):
        """
        绘制当前状态
        
        Args:
            robot_position: 机器人位置
            hand_position: 手部位置
            fixed_point: 固定点位置
            trajectory_points: 轨迹点列表
            blocking_point: 阻塞点位置（可选）
        """
        self.canvas.fill((255, 255, 255))
        self.draw_trajectory(trajectory_points)
        self.draw_robot(robot_position)
        self.draw_hand(hand_position)
        
        if blocking_point is not None:
            pygame.draw.circle(
                self.canvas, 
                (255, 255, 0), 
                (int(blocking_point[0] * self.cell_size), 
                 int(blocking_point[1] * self.cell_size)), 
                int(self.cell_size * 0.2), 
                5
            )

        self.window.blit(self.canvas, (0, 0))
        pygame.display.flip()

    def draw_trajectory(self, trajectory_points):
        """绘制轨迹"""
        if len(trajectory_points) > 1:
            scaled_points = [
                (int(point[0] * self.cell_size), int(point[1] * self.cell_size)) 
                for point in trajectory_points
            ]
            pygame.draw.lines(self.canvas, (0, 0, 255), False, scaled_points, 2)
            for point_coord in scaled_points:
                pygame.draw.circle(self.canvas, (0, 0, 255), point_coord, 3)

    def draw_robot(self, robot_position):
        """绘制机器人"""
        pygame.draw.circle(
            self.canvas, 
            (255, 0, 0), 
            (int(robot_position[0] * self.cell_size), 
             int(robot_position[1] * self.cell_size)), 
            int(self.cell_size * 0.2)
        )

    def draw_hand(self, hand_position):
        """绘制手部"""
        pygame.draw.circle(
            self.canvas, 
            (0, 255, 0), 
            (int(hand_position[0] * self.cell_size), 
             int(hand_position[1] * self.cell_size)), 
            int(self.cell_size * 0.2)
        )
        self.canvas.blit(
            pygame.transform.scale(
                self.virus_image, 
                (int(self.cell_size * 2), int(self.cell_size * 2))
            ), 
            (int((hand_position[0] - 1) * self.cell_size), 
             int((hand_position[1] - 1) * self.cell_size))
        )

    def quit(self):
        """清理资源"""
        pygame.quit()

