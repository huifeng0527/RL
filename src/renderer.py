"""Academic-style rendering for the rehabilitation environment."""

import math
import pygame
import pygame.gfxdraw
import numpy as np


# --- Academic Style Color Palette ---
COLORS = {
    'bg_main': (250, 250, 250),
    'grid': (230, 230, 230),
    'arm_fill': (255, 204, 188),
    'arm_border': (230, 74, 25),
    'robot': (220, 50, 50),
    'robot_shadow': (200, 50, 50),
    'hand_core': (46, 204, 113),
    'trajectory': (52, 152, 219),
    'text': (50, 60, 80)
}


def draw_aa_circle(surf, color, center, radius):
    """Draw anti-aliased filled circle."""
    x, y = int(center[0]), int(center[1])
    pygame.gfxdraw.aacircle(surf, x, y, radius, color)
    pygame.gfxdraw.filled_circle(surf, x, y, radius, color)


def draw_capsule(surf, color, start_pos, end_pos, width):
    """Draw capsule shape (for arm simulation)."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return

    angle = np.arctan2(y2 - y1, x2 - x1)

    dx = width / 2 * np.sin(angle)
    dy = width / 2 * np.cos(angle)

    points = [
        (x1 - dx, y1 + dy),
        (x2 - dx, y2 + dy),
        (x2 + dx, y2 - dy),
        (x1 + dx, y1 - dy)
    ]
    pygame.gfxdraw.aapolygon(surf, points, color)
    pygame.gfxdraw.filled_polygon(surf, points, color)

    draw_aa_circle(surf, color, start_pos, int(width / 2))
    draw_aa_circle(surf, color, end_pos, int(width / 2))


def draw_capsule_rotated(surf, color, center_x, center_y, width, length, angle_deg):
    """Draw a rotated capsule."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    x2 = center_x + length * cos_a
    y2 = center_y + length * sin_a

    dx = (width / 2) * math.sin(rad)
    dy = (width / 2) * math.cos(rad)

    points = [
        (center_x - dx, center_y + dy),
        (center_x + dx, center_y - dy),
        (x2 + dx, y2 - dy),
        (x2 - dx, y2 + dy)
    ]

    pygame.gfxdraw.aapolygon(surf, points, color)
    pygame.gfxdraw.filled_polygon(surf, points, color)

    pygame.gfxdraw.aacircle(surf, int(center_x), int(center_y), int(width / 2), color)
    pygame.gfxdraw.filled_circle(surf, int(center_x), int(center_y), int(width / 2), color)
    pygame.gfxdraw.aacircle(surf, int(x2), int(y2), int(width / 2), color)
    pygame.gfxdraw.filled_circle(surf, int(x2), int(y2), int(width / 2), color)


def draw_nail(start_x, start_y, width, length, angle, fill_color, nail_color):
    """Draw a nail on a finger."""
    nail_dist = length * 0.75
    nail_w = width * 0.6
    nail_h = width * 0.5

    rad = math.radians(angle)
    nx = start_x + nail_dist * math.cos(rad)
    ny = start_y + nail_dist * math.sin(rad)

    draw_capsule_rotated(surf, nail_color, nx, ny, nail_h, nail_w * 0.2, angle)


def draw_detailed_hand(surf, fill_color, border_color, center, radius, angle_deg=0):
    """Draw a detailed hand with fingers, nails, and knuckles."""
    cx, cy = center
    base_rad = math.radians(angle_deg)

    nail_color = (min(fill_color[0] + 20, 255),
                  min(fill_color[1] + 20, 255),
                  min(fill_color[2] + 20, 255))
    knuckle_color = border_color

    def get_rotated_offset(ox, oy):
        rx = ox * math.cos(base_rad) - oy * math.sin(base_rad)
        ry = ox * math.sin(base_rad) + oy * math.cos(base_rad)
        return cx + rx, cy + ry

    palm_size = radius * 1.15
    finger_width = radius * 0.48
    finger_len = radius * 1.5

    fingers = [
        (radius * 0.15, -radius * 0.55, -12, 0.9),
        (radius * 0.25, 0, 0, 1.0),
        (radius * 0.15, radius * 0.55, 12, 0.9),
        (radius * 0.05, radius * 1.0, 25, 0.75)
    ]

    for fx, fy, fang, flen in fingers:
        pos = get_rotated_offset(fx, fy)
        draw_capsule_rotated(surf, border_color, pos[0], pos[1], finger_width + 4, finger_len * flen, angle_deg + fang)

    thumb_pos = get_rotated_offset(-radius * 0.2, -radius * 0.6)
    thumb_angle = -55
    draw_capsule_rotated(surf, border_color, thumb_pos[0], thumb_pos[1], finger_width * 1.3 + 4, finger_len * 0.85, angle_deg + thumb_angle)

    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(palm_size), border_color)
    pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), int(palm_size), border_color)

    for fx, fy, fang, flen in fingers:
        pos = get_rotated_offset(fx, fy)
        current_len = finger_len * flen
        current_angle = angle_deg + fang

        draw_capsule_rotated(surf, fill_color, pos[0], pos[1], finger_width, current_len, current_angle)
        draw_nail(pos[0], pos[1], finger_width, current_len, current_angle, fill_color, nail_color)

        pygame.gfxdraw.aacircle(surf, int(pos[0]), int(pos[1]), int(finger_width * 0.4), (230, 150, 130))
        pygame.gfxdraw.filled_circle(surf, int(pos[0]), int(pos[1]), int(finger_width * 0.4), (230, 150, 130))

    draw_capsule_rotated(surf, fill_color, thumb_pos[0], thumb_pos[1], finger_width * 1.3, finger_len * 0.85, angle_deg + thumb_angle)
    draw_nail(thumb_pos[0], thumb_pos[1], finger_width * 1.3, finger_len * 0.85, angle_deg + thumb_angle, fill_color, nail_color)

    pygame.gfxdraw.aacircle(surf, int(cx), int(cy), int(palm_size - 2), fill_color)
    pygame.gfxdraw.filled_circle(surf, int(cx), int(cy), int(palm_size - 2), fill_color)

    for i in [-1, 1]:
        start = get_rotated_offset(-radius * 0.5, i * radius * 0.3)
        end = get_rotated_offset(0, i * radius * 0.2)
        pygame.draw.line(surf, (230, 150, 130), start, end, 2)


def render_aesthetic(robot_pos, hand_pos, fix_point, trajectory_points,
                    grid_size=10, cell_size=50, window=None):
    """Render the rehabilitation environment with academic style."""
    width_px = int(grid_size * cell_size * 1.5)
    height_px = int(grid_size * cell_size)

    if window is None:
        if not pygame.get_init():
            pygame.init()
        window = pygame.display.set_mode((width_px, height_px))

    canvas = pygame.Surface((width_px, height_px))
    canvas.fill(COLORS['bg_main'])

    for x in range(0, width_px, cell_size):
        pygame.draw.line(canvas, COLORS['grid'], (x, 0), (x, height_px), 1)
    for y in range(0, height_px, cell_size):
        pygame.draw.line(canvas, COLORS['grid'], (0, y), (width_px, y), 1)

    fix_px = np.array(fix_point) * cell_size
    hand_px = np.array(hand_pos) * cell_size

    arm_width = 40
    draw_capsule(canvas, COLORS['arm_border'], fix_px, hand_px, arm_width + 4)
    draw_capsule(canvas, COLORS['arm_fill'], fix_px, hand_px, arm_width)

    if len(trajectory_points) > 1:
        recent_points = trajectory_points[-50:]

        for i in range(len(recent_points) - 1):
            pt1 = (int(recent_points[i][0] * cell_size), int(recent_points[i][1] * cell_size))
            pt2 = (int(recent_points[i + 1][0] * cell_size), int(recent_points[i + 1][1] * cell_size))

            if i > len(recent_points) - 10:
                pygame.draw.line(canvas, COLORS['trajectory'], pt1, pt2, 2)

            draw_aa_circle(canvas, COLORS['trajectory'], pt1, 2)

    robot_px = np.array(robot_pos) * cell_size
    robot_radius = int(cell_size * 0.25)

    shadow_offset = (3, 3)
    draw_aa_circle(canvas, (200, 200, 200), robot_px + shadow_offset, robot_radius)

    draw_aa_circle(canvas, COLORS['robot'], robot_px, robot_radius)
    draw_aa_circle(canvas, (255, 255, 255), robot_px - (robot_radius * 0.3, robot_radius * 0.3), int(robot_radius * 0.3))

    dx = hand_px[0] - fix_px[0]
    dy = hand_px[1] - fix_px[1]
    hand_angle = math.degrees(math.atan2(dy, dx))

    hand_size = cell_size * 0.6

    draw_detailed_hand(canvas, COLORS['arm_fill'], COLORS['arm_border'],
                       hand_px, hand_size, angle_deg=hand_angle)

    draw_aa_circle(canvas, (255, 255, 255), hand_px, int(cell_size * 0.08))

    window.blit(canvas, (0, 0))
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            import sys
            sys.exit()

    return canvas
