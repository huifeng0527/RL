"""
Auto Homography Calibration via YOLO
======================================
全自动标定：机械臂移动到多个位置，YOLO 检测marker中心，
自动收集 world→pixel 点对应，计算单应性矩阵并保存。

用法:
    python auto_calibrate_homography.py --samples 12 --dx 0.1 --dy 0.1
    python auto_calibrate_homography.py --interactive   # 逐个确认后采集
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time

_rl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _rl_root)

from ultralytics import YOLO
from robot_control.ur_control import URControl
from cv.get_workspace import get_workspace

# ========================================================
# 配置参数
# ========================================================
desired_width = 2592
desired_height = 1944

CALI_DIR = os.path.join(_rl_root, "camera_calibration")
CALIB_DATA_PATH = os.path.join(CALI_DIR, "calibration_data.npz")
HOMOGRAPHY_SAVE_PATH = os.path.join(CALI_DIR, "Homography_matrix.npy")
YOLO_MODEL_PATH = os.path.join(_rl_root, "runs", "detect", "train3", "weights", "best.onnx")

# 默认 YOLO 置信度
DEFAULT_CONF = 0.5


# ========================================================
# 辅助函数
# ========================================================
def load_calibration():
    """加载内参"""
    data = np.load(CALIB_DATA_PATH)
    K = data['K']
    dist = data['dist_coeffs']
    new_cam, roi = cv2.getOptimalNewCameraMatrix(K, dist, (desired_width, desired_height), 0, (desired_width, desired_height))
    return K, dist, new_cam


def detect_marker(frame, model, conf=0.5):
    """
    用 YOLO 检测单目marker，返回像素中心 [cx, cy]。
    如果检测到多个或零个则抛出异常。
    """
    results = model.predict(frame, conf=conf, save=False, imgsz=frame.shape[1::-1], verbose=False)
    if not results or len(results[0].boxes) == 0:
        raise RuntimeError("YOLO: 未检测到任何 marker")
    if len(results[0].boxes) > 1:
        raise RuntimeError(f"YOLO: 检测到 {len(results[0].boxes)} 个目标，预期仅1个")

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return cx, cy, (x1, y1, x2, y2)


def preprocess_frame(frame, K, dist, new_camera_matrix):
    """
    畸变矫正 + 裁剪 + 旋转（与 eval.py 保持一致）
    """
    undist = cv2.undistort(frame, K, dist, None, new_camera_matrix)
    undist = get_workspace(undist)                          # [:, 600:-400]
    undist = cv2.rotate(undist, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return undist


def draw_detection(frame, cx, cy, box, label=""):
    """在帧上绘制检测结果"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(frame, f'({cx}, {cy}) {label}', (cx + 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def compute_homography(object_pts, img_pts):
    """计算单应性矩阵并做基础校验"""
    obj_np = np.array(object_pts, dtype=np.float32)
    img_np = np.array(img_pts, dtype=np.float32)

    H, mask = cv2.findHomography(obj_np, img_np)
    if H is None:
        raise RuntimeError("单应性矩阵计算失败，点可能共线或数量不足")

    # 计算重投影误差
    img_pts_reproj = cv2.perspectiveTransform(obj_np.reshape(-1, 1, 2), H)
    errors = np.linalg.norm(img_pts_reproj.squeeze() - img_np, axis=1)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"  重投影误差 — mean: {mean_err:.2f}px, max: {max_err:.2f}px")

    return H, mean_err, max_err


# ========================================================
# 主流程
# ========================================================
def auto_calibrate(
    robot_ip='192.168.1.2',
    num_samples=12,
    dx_range=0.1,
    dy_range=0.1,
    wait_time=2.0,
    conf=0.5,
    interactive=False,
    preview=True
):
    """
    自动标定流程：
    1. 连接机械臂和相机
    2. 移动到 num_samples 个随机位置，同时记录 world 坐标
    3. YOLO 检测 marker 像素位置
    4. 计算并保存 Homography 矩阵
    """
    print("[1/5] 加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL_PATH)

    print("[2/5] 连接机械臂...")
    ur = URControl(robot_ip)

    print("[3/5] 加载相机内参...")
    K, dist, new_cam = load_calibration()

    print("[4/5] 初始化相机...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    if not cap.isOpened():
        raise RuntimeError("相机无法打开")

    # 获取中心点（基准位置）
    center_pose = ur.get_robot_pose()
    print(f"  机械臂中心点: {center_pose[:2]}")

    # 数据容器
    object_pts = []   # world coords (x, y)
    img_pts = []     # pixel coords (cx, cy)

    cv2.namedWindow("Auto Calibration", cv2.WINDOW_NORMAL)

    for i in range(num_samples):
        # 随机生成目标位置（相对于中心）
        dx = np.random.uniform(-dx_range, dx_range)
        dy = np.random.uniform(-dy_range, dy_range)
        pose = ur.get_robot_pose()
        target_pose = [pose[0] + dx, pose[1] + dy, pose[2], pose[3], pose[4], pose[5]]

        print(f"\n--- Sample {i+1}/{num_samples} ---")
        print(f"  目标世界坐标: ({target_pose[0]:.4f}, {target_pose[1]:.4f})")

        # 移动机械臂
        ur.move_robot(target_pose)
        time.sleep(wait_time)

        # 读取并预处理图像
        ret, frame = cap.read()
        if not ret:
            print("  [警告] 无法读取帧，跳过此样本")
            ur.move_robot(center_pose)
            time.sleep(wait_time)
            continue

        undist = preprocess_frame(frame, K, dist, new_cam)

        # YOLO 检测
        try:
            cx, cy, box = detect_marker(undist, model, conf)
            print(f"  检测到像素坐标: ({cx}, {cy})")
        except RuntimeError as e:
            print(f"  [警告] {e}，跳过此样本")
            ur.move_robot(center_pose)
            time.sleep(wait_time)
            continue

        # 预览
        display = undist.copy()
        draw_detection(display, cx, cy, box, f"#{i+1}")
        cv2.putText(display, f"Sample {i+1}/{num_samples} | World: ({target_pose[0]:.3f}, {target_pose[1]:.3f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Auto Calibration", display)
        cv2.waitKey(300 if not interactive else 1)

        # 交互模式：按键继续
        if interactive:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n用户中断，放弃本次采集")
                break
            elif key == ord('s'):
                print("  跳过此样本")
                ur.move_robot(center_pose)
                time.sleep(wait_time)
                continue

        # 记录数据
        object_pts.append([target_pose[0], target_pose[1]])
        img_pts.append([cx, cy])

        print(f"  已采集: {len(object_pts)} 对点")

        # 返回中心点（休息一下）
        ur.move_robot(center_pose)
        time.sleep(wait_time)

        if preview:
            cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()

    # ========================================================
    # 计算 Homography
    # ========================================================
    print(f"\n[5/5] 计算 Homography（{len(object_pts)} 对点）...")

    if len(object_pts) < 4:
        raise RuntimeError(f"有效点数量不足（{len(object_pts)}），至少需要4对")

    H, mean_err, max_err = compute_homography(object_pts, img_pts)

    # 保存
    np.save(HOMOGRAPHY_SAVE_PATH, H)
    print(f"  Homography 矩阵已保存至: {HOMOGRAPHY_SAVE_PATH}")

    # 打印结果摘要
    print("\n========== 标定结果 ==========")
    print(f"有效样本数: {len(object_pts)}")
    print(f"重投影误差 — mean: {mean_err:.2f}px, max: {max_err:.2f}px")
    print(f"Homography 矩阵:\n{H}")
    print("==============================")

    return H, object_pts, img_pts


# ========================================================
# 命令行入口
# ========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto Homography Calibration via YOLO")
    parser.add_argument('--ip', default='192.168.1.2', help='机械臂 IP')
    parser.add_argument('--samples', '-n', type=int, default=12, help='采集样本数量（默认12）')
    parser.add_argument('--dx', type=float, default=0.1, help='X方向移动范围（米，默认0.1）')
    parser.add_argument('--dy', type=float, default=0.1, help='Y方向移动范围（米，默认0.1）')
    parser.add_argument('--wait', type=float, default=2.0, help='机械臂到位等待时间（秒，默认2.0）')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO 置信度阈值（默认0.5）')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互模式（逐帧确认）')
    args = parser.parse_args()

    try:
        auto_calibrate(
            robot_ip=args.ip,
            num_samples=args.samples,
            dx_range=args.dx,
            dy_range=args.dy,
            wait_time=args.wait,
            conf=args.conf,
            interactive=args.interactive
        )
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)