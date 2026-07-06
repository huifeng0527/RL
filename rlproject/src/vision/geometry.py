from pathlib import Path

import cv2
import numpy as np


def as_homogeneous_point(point):
    coords = np.asarray(point, dtype=np.float32).flatten()
    return np.array([coords[0], coords[1], 1], dtype=np.float32)


def pixel_to_world(H, pixel_coords):
    point = np.linalg.inv(H) @ as_homogeneous_point(pixel_coords)
    point /= point[2]
    return point[:2]


def world_to_pixel(H, world_coords):
    point = H @ as_homogeneous_point(world_coords)
    point /= point[2]
    return point[:2]


def compute_homography(world_points, image_points):
    world = np.asarray(world_points, dtype=np.float32)
    image = np.asarray(image_points, dtype=np.float32)
    H, status = cv2.findHomography(world, image)
    if H is None:
        raise RuntimeError("Homography calculation failed")
    return H, status


def reprojection_error(H, world_points, image_points):
    world = np.asarray(world_points, dtype=np.float32)
    image = np.asarray(image_points, dtype=np.float32)
    projected = cv2.perspectiveTransform(world.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - image, axis=1)
    return float(np.mean(errors)), float(np.max(errors)), errors


def save_homography(path, H):
    np.save(Path(path), H)


def load_homography(path):
    return np.load(Path(path))
