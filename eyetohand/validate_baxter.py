#!/usr/bin/env python3
"""Read-only validation of a fixed-camera Baxter calibration on new board/arm poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from accurate_eye_to_hand import (
    build_object_points,
    detect_chessboard,
    invert_transform,
    load_robot_poses,
    numeric_image_files,
    rotation_angle_deg,
)


def matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f'{name} 必须是有限数值的 4×4 变换矩阵')
    return result


def validate(dataset: Path, calibration: Path) -> int:
    if dataset.resolve() == calibration.parent.resolve():
        raise ValueError('验证数据必须是新采集的目录，不能用原标定目录验证自身')
    result = json.loads(calibration.read_text(encoding='utf-8'))
    if 'T_gripper_target' not in result:
        raise ValueError('标定结果缺少 T_gripper_target，请用更新后的 accurate_eye_to_hand.py 重新求解 1/')
    T_base_camera = matrix(result['T_base_camera'], 'T_base_camera')
    T_gripper_target = matrix(result['T_gripper_target'], 'T_gripper_target')
    intr = json.loads((dataset / 'intrinsics.json').read_text(encoding='utf-8'))
    K = np.asarray(intr['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(intr.get('dist_coeffs', [0] * 5), dtype=np.float64).reshape(-1)
    if not np.allclose(K, result['camera_matrix'], atol=1e-4, rtol=0) or not np.allclose(
            dist, result['dist_coeffs'], atol=1e-6, rtol=0):
        raise ValueError('新数据的相机内参与标定时不同；检查是否换了相机/分辨率/矫正方式')
    session = json.loads((dataset / 'session.json').read_text(encoding='utf-8'))
    if session.get('arm') != 'right':
        raise ValueError('这份结果由右臂标定，新数据也必须采集右臂')
    poses = load_robot_poses(dataset / 'record.txt', session.get('euler_order', 'xyz'))
    cols, rows = int(result['chessboard_cols']), int(result['chessboard_rows'])
    pattern = (cols, rows)
    object_points = build_object_points(pattern, float(result['square_size_m']))
    images = numeric_image_files(dataset / 'picture')
    if not images:
        raise ValueError(f'{dataset}/picture 中没有数字命名的图片')
    errors_mm, angles_deg, pnp_px = [], [], []
    skipped = []
    print('图片     PnP误差(px)    棋盘位置误差(mm)    棋盘角度误差(°)')
    for image in images:
        idx = int(image.stem)
        if idx >= len(poses):
            skipped.append(f'{image.name}: 找不到对应的 record.txt 行')
            continue
        ok, size, corners = detect_chessboard(image, pattern)
        if size != (intr['width'], intr['height']) or not ok or corners is None:
            skipped.append(f'{image.name}: 图片分辨率不符或棋盘检测失败')
            continue
        points = corners.reshape(-1, 2)
        success, rvec, tvec = cv2.solvePnP(object_points, points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            skipped.append(f'{image.name}: solvePnP 失败')
            continue
        reproj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        pixel_rms = float(np.sqrt(np.mean((reproj.reshape(-1, 2) - points) ** 2)))
        R, _ = cv2.Rodrigues(rvec)
        T_camera_target = np.eye(4)
        T_camera_target[:3, :3] = R
        T_camera_target[:3, 3] = tvec.reshape(3)
        # The same board location is predicted independently from the camera and robot endpoint.
        observed = T_base_camera @ T_camera_target
        predicted = poses[idx] @ T_gripper_target
        delta = invert_transform(predicted) @ observed
        position_mm = float(np.linalg.norm(delta[:3, 3]) * 1000)
        angle_deg = rotation_angle_deg(np.eye(3), delta[:3, :3])
        errors_mm.append(position_mm)
        angles_deg.append(angle_deg)
        pnp_px.append(pixel_rms)
        print(f'{image.name:>7} {pixel_rms:>13.3f} {position_mm:>19.2f} {angle_deg:>20.2f}')
    for reason in skipped:
        print('[跳过]', reason)
    if not errors_mm:
        raise ValueError('没有有效验证图片')
    print(f'有效: {len(errors_mm)}/{len(images)}；棋盘位置误差平均 {np.mean(errors_mm):.2f} mm、'
          f'最大 {max(errors_mm):.2f} mm；角度误差平均 {np.mean(angles_deg):.2f}°、'
          f'最大 {max(angles_deg):.2f}°；单图 PnP 均方根平均 {np.mean(pnp_px):.3f} px。')
    print('这些误差检验新姿态下相机/末端/棋盘是否一致；真实夹爪中心与深度测量仍需单独验证。')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='新采集姿态的 Baxter 眼在手外标定只读验证')
    parser.add_argument('--dataset', type=Path, required=True, help='新采集目录，例如 2')
    parser.add_argument('--calibration', type=Path, required=True, help='旧数据求得的 1/hand_eye_result.json')
    args = parser.parse_args()
    try:
        return validate(args.dataset, args.calibration)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
