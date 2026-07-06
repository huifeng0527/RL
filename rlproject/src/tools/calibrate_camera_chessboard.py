import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np

from vision.calibration_utils import DEFAULT_IMAGE_SIZE, build_new_camera_matrix, save_calibration_data
from vision.paths import DEFAULT_CALIBRATION_DATA_PATH


def collect_chessboard_points(image_paths, checkerboard, square_size, preview=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    image_shape = None

    if preview:
        cv2.namedWindow("Chessboard Calibration", cv2.WINDOW_NORMAL)

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: could not read {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print(f"Warning: could not find chessboard corners in {image_path}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)

        if preview:
            display = img.copy()
            cv2.drawChessboardCorners(display, checkerboard, corners2, found)
            cv2.imshow("Chessboard Calibration", display)
            if cv2.waitKey(300) & 0xFF == ord("q"):
                break

    if preview:
        cv2.destroyWindow("Chessboard Calibration")

    return objpoints, imgpoints, image_shape


def calibrate_camera(image_dir, output, checkerboard=(6, 9), square_size=24.0, image_size=DEFAULT_IMAGE_SIZE, preview=False):
    image_dir = Path(image_dir)
    image_paths = sorted(
        path for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for path in image_dir.glob(suffix)
    )
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {image_dir}")

    objpoints, imgpoints, detected_image_shape = collect_chessboard_points(
        image_paths,
        checkerboard=checkerboard,
        square_size=square_size,
        preview=preview,
    )
    if not objpoints:
        raise RuntimeError("No valid chessboard detections found")

    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        detected_image_shape,
        None,
        None,
    )
    new_camera_matrix, _ = build_new_camera_matrix(K, dist_coeffs, image_size=image_size, alpha=0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_data(output, ret, K, dist_coeffs, rvecs, tvecs, new_camera_matrix)

    print("Calibration successful")
    print(f"Images used: {len(objpoints)} / {len(image_paths)}")
    print(f"Camera matrix:\n{K}")
    print(f"Distortion coefficients:\n{dist_coeffs}")
    print(f"Saved calibration data to: {output}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images")
    parser.add_argument("--image-dir", required=True, help="Directory containing chessboard images")
    parser.add_argument("--output", default=str(DEFAULT_CALIBRATION_DATA_PATH), help="Output .npz path")
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(6, 9), metavar=("ROWS", "COLS"))
    parser.add_argument("--square-size", type=float, default=24.0, help="Chessboard square size in the chosen world unit")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    calibrate_camera(
        image_dir=args.image_dir,
        output=args.output,
        checkerboard=tuple(args.checkerboard),
        square_size=args.square_size,
        image_size=(args.width, args.height),
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
