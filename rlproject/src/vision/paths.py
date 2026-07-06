from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
CAMERA_CALIBRATION_DIR = SRC_ROOT / "camera_calibration"
DEFAULT_CALIBRATION_DATA_PATH = CAMERA_CALIBRATION_DIR / "calibration_data.npz"
DEFAULT_HOMOGRAPHY_PATH = CAMERA_CALIBRATION_DIR / "Homography_matrix.npy"
DEFAULT_YOLO_MODEL_PATH = SRC_ROOT / "runs" / "detect" / "train3" / "weights" / "best.onnx"
