import cv2

from vision.calibration_utils import DEFAULT_IMAGE_SIZE, build_new_camera_matrix

DEFAULT_WORKSPACE_CROP_LEFT = 600
DEFAULT_WORKSPACE_CROP_RIGHT = 400
DEFAULT_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE


def undistort_frame(frame, K, dist_coeffs, new_camera_matrix):
    return cv2.undistort(frame, K, dist_coeffs, None, new_camera_matrix)


def crop_workspace(frame, left=DEFAULT_WORKSPACE_CROP_LEFT, right=DEFAULT_WORKSPACE_CROP_RIGHT):
    if right == 0:
        return frame[:, left:]
    return frame[:, left:-right]


def rotate_workspace(frame, rotation=DEFAULT_ROTATION):
    return cv2.rotate(frame, rotation)


def preprocess_workspace_frame(
    frame,
    K,
    dist_coeffs,
    new_camera_matrix=None,
    image_size=DEFAULT_IMAGE_SIZE,
    crop_left=DEFAULT_WORKSPACE_CROP_LEFT,
    crop_right=DEFAULT_WORKSPACE_CROP_RIGHT,
    rotation=DEFAULT_ROTATION,
):
    if new_camera_matrix is None:
        new_camera_matrix, _ = build_new_camera_matrix(K, dist_coeffs, image_size=image_size, alpha=0)
    workspace = undistort_frame(frame, K, dist_coeffs, new_camera_matrix)
    workspace = crop_workspace(workspace, left=crop_left, right=crop_right)
    if rotation is not None:
        workspace = rotate_workspace(workspace, rotation=rotation)
    return workspace
