import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np

from vision.calibration_utils import DEFAULT_IMAGE_SIZE, load_calibration_data
from vision.geometry import compute_homography, reprojection_error, save_homography
from vision.paths import DEFAULT_CALIBRATION_DATA_PATH, DEFAULT_HOMOGRAPHY_PATH
from vision.preprocessing import preprocess_workspace_frame


def run_manual_homography_calibration(
    robot_ip="192.168.1.2",
    calibration_data_path=DEFAULT_CALIBRATION_DATA_PATH,
    output_path=DEFAULT_HOMOGRAPHY_PATH,
    camera_index=0,
    image_size=DEFAULT_IMAGE_SIZE,
    min_points=8,
):
    from robot_control.ur_control import URControl

    calibration = load_calibration_data(calibration_data_path, image_size=image_size, alpha=0)
    robot = URControl(robot_ip)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size[1])
    if not cap.isOpened():
        robot.disconnect()
        raise RuntimeError("Camera is not accessible")

    window_name = "Manual Homography Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    world_points = []
    image_points = []
    latest_frame = None

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pose = robot.get_robot_pose()
        world_points.append([pose[0], pose[1]])
        image_points.append([x, y])
        print(f"Point {len(world_points)}: world=({pose[0]:.6f}, {pose[1]:.6f}), pixel=({x}, {y})")
        if len(world_points) >= min_points:
            H, _ = compute_homography(world_points, image_points)
            mean_err, max_err, _ = reprojection_error(H, world_points, image_points)
            save_homography(output_path, H)
            print(f"Saved homography to: {output_path}")
            print(f"Reprojection error: mean={mean_err:.2f}px, max={max_err:.2f}px")
            print(H)

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        print(f"Connected to robot: {robot_ip}")
        print(f"Left-click marker positions. Press q to finish. Minimum points: {min_points}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: could not read camera frame")
                continue

            latest_frame = preprocess_workspace_frame(
                frame,
                calibration.K,
                calibration.dist_coeffs,
                calibration.new_camera_matrix,
                image_size=image_size,
            )
            display = latest_frame.copy()
            for idx, point in enumerate(image_points, start=1):
                cv2.circle(display, tuple(map(int, point)), 8, (255, 0, 255), -1)
                cv2.putText(display, str(idx), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"points: {len(image_points)}/{min_points}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(window_name, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        robot.disconnect()

    if len(world_points) < 4:
        raise RuntimeError(f"Need at least 4 points, got {len(world_points)}")

    H, _ = compute_homography(world_points, image_points)
    mean_err, max_err, _ = reprojection_error(H, world_points, image_points)
    save_homography(output_path, H)
    print(f"Final homography saved to: {output_path}")
    print(f"Final reprojection error: mean={mean_err:.2f}px, max={max_err:.2f}px")
    return H, world_points, image_points


def main():
    parser = argparse.ArgumentParser(description="Manually calibrate world-to-pixel homography with robot pose and mouse clicks")
    parser.add_argument("--ip", default="192.168.1.2", help="UR robot IP")
    parser.add_argument("--calibration-data", default=str(DEFAULT_CALIBRATION_DATA_PATH), help="Camera calibration .npz path")
    parser.add_argument("--output", default=str(DEFAULT_HOMOGRAPHY_PATH), help="Output Homography_matrix.npy path")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--min-points", type=int, default=8)
    args = parser.parse_args()

    run_manual_homography_calibration(
        robot_ip=args.ip,
        calibration_data_path=Path(args.calibration_data),
        output_path=Path(args.output),
        camera_index=args.camera_index,
        image_size=(args.width, args.height),
        min_points=args.min_points,
    )


if __name__ == "__main__":
    main()
