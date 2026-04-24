import pygame

class EnvironmentRenderer:
    def __init__(self, grid_size=10, cell_size=50, window_size=None):
        # 初始化窗口和基本的绘图元素
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = window_size or (grid_size * cell_size * 1.5, grid_size * cell_size)
        self.window = pygame.display.set_mode(self.window_size)
        self.canvas = pygame.Surface(self.window_size)
        self.canvas.fill((255, 255, 255))  # 白色背景

        # 预加载需要的图像
        self.virus_image = pygame.image.load("hand.png").convert_alpha()  # 这可以用来绘制手
        self.font = pygame.font.Font(None, 24)

    def render(self, robot_position, hand_position,fixed_point, trajectory_points,blocking_point):


        """ 绘制当前状态到窗口 """


        # print(blocking_point,robot_position)
        self.canvas.fill((255, 255, 255))  # 清空画布
        self.draw_trajectory(trajectory_points)
        self.draw_robot(robot_position)
        self.draw_hand(hand_position)
        # self.draw_line_from_hand_to_center(hand_position,fixed_point)
        # self.draw_text(hand_position)
        pygame.draw.circle(self.canvas, (255, 255, 0), (int(blocking_point[0] * self.cell_size), int(blocking_point[1] * self.cell_size)), int(self.cell_size * 0.2),5)

        self.window.blit(self.canvas, (0, 0))
        pygame.display.flip()  # 更新屏幕

    def draw_trajectory(self, trajectory_points):
        """ 绘制轨迹 """
        if len(trajectory_points) > 1:
            scaled_points = [(int(point[0] * self.cell_size), int(point[1] * self.cell_size)) for point in trajectory_points]
            pygame.draw.lines(self.canvas, (0, 0, 255), False, scaled_points, 2)
            for point_coord in scaled_points:
                pygame.draw.circle(self.canvas, (0, 0, 255), point_coord, 3)

    def draw_robot(self, robot_position):
        """ 绘制机器人的位置 """
        pygame.draw.circle(self.canvas, (255, 0, 0), (int(robot_position[0] * self.cell_size), int(robot_position[1] * self.cell_size)), int(self.cell_size * 0.2))

    def draw_hand(self, hand_position):
        """ 绘制手的位置 """
        pygame.draw.circle(self.canvas, (0, 255, 0), (int(hand_position[0] * self.cell_size), int(hand_position[1] * self.cell_size)), int(self.cell_size * 0.2))
        self.canvas.blit(pygame.transform.scale(self.virus_image, (int(self.cell_size * 2), int(self.cell_size * 2))), 
                         (int((hand_position[0]-1) * self.cell_size), int((hand_position[1]-1) * self.cell_size)))

    def draw_line_from_hand_to_center(self, hand_position,fixed_point):
        """ 从手的位置绘制一条线到某个中心 """
        pygame.draw.lines(self.canvas, (255, 224, 189), False, 
                          [[x.item() * self.cell_size for x in hand_position], [ self.cell_size * fixed_point[0],  self.cell_size * fixed_point[1]]], width=25)

    def draw_text(self, hand_position):
        """ 绘制手的位置文本 """
        text = self.font.render(f"{hand_position[0]},{hand_position[1]}", True, (0, 0, 255))
        text_rect = text.get_rect()
        text_rect.center = (int(hand_position[0] * self.cell_size), int(hand_position[1] * self.cell_size))
        self.window.blit(self.canvas, self.canvas.get_rect())
        self.window.blit(text, text_rect)

    def quit(self):
        """ 清理工作 """
        pygame.quit()