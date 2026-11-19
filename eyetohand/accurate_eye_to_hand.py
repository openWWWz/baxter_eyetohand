#!/usr/bin/env python3
# coding: utf-8
"""
Eye-to-hand calibration for a fixed camera and a chessboard attached to the robot hand.

Output convention:
    T_base_camera maps a 3D point from camera coordinates to robot base coordinates.
    P_base = R_base_camera @ P_camera + t_base_camera
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


np.set_printoptions(precision=8, suppress=True)


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class ViewData:
    image_id: int
    image_path: Path
    image_points: np.ndarray
    object_points: np.ndarray
    T_base_gripper: np.ndarray
    T_camera_target: np.ndarray | None = None


@dataclass
class CalibrationResult:
    name: str
    T_base_camera: np.ndarray
    T_gripper_target: np.ndarray
    board_rot_mean_deg: float
    board_rot_max_deg: float
    board_trans_mean_m: float
    board_trans_max_m: float
    reproj_rms_px: float


def make_transform(R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R_mat, dtype=np.float64)
    T[:3, 3] = np.asarray(t_vec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4, dtype=np.float64)
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t_vec
    return T_inv


def transform_to_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3].copy()


def rt_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return make_transform(Rotation.from_rotvec(rvec).as_matrix(), tvec)


def normalize_quat_sign(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).copy()
    if quat_xyzw[3] < 0:
        quat_xyzw *= -1.0
    return quat_xyzw


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    value = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def average_transform(transforms: list[np.ndarray]) -> np.ndarray:
    translations = np.array([T[:3, 3] for T in transforms])
    quats = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in transforms])
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] *= -1.0
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    return make_transform(Rotation.from_quat(mean_quat).as_matrix(), np.mean(translations, axis=0))


def build_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(square_size)
    return obj


def load_robot_poses(record_file: Path, euler_order: str) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for line_no, raw_line in enumerate(record_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
        if len(parts) != 6:
            raise ValueError(f"{record_file}:{line_no} should contain 6 values, got {len(parts)}")
        x, y, z, rx, ry, rz = map(float, parts)
        R_base_gripper = Rotation.from_euler(euler_order, [rx, ry, rz], degrees=False).as_matrix()
        poses.append(make_transform(R_base_gripper, np.array([x, y, z], dtype=np.float64)))
    return poses


def numeric_image_files(image_dir: Path) -> list[Path]:
    files = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(image_dir.glob(suffix))
    numeric = [p for p in files if p.stem.isdigit()]
    return sorted(numeric, key=lambda p: int(p.stem))


def detect_chessboard(
    image_path: Path,
    pattern_size: tuple[int, int],
    use_sb_first: bool = True,
) -> tuple[bool, tuple[int, int] | None, np.ndarray | None]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    if use_sb_first and hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if ok:
            return True, image_size, corners.astype(np.float32)

    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, image_size, corners.astype(np.float32)

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if ok:
            return True, image_size, corners.astype(np.float32)

    return False, image_size, None


def load_views(args: argparse.Namespace) -> tuple[list[ViewData], tuple[int, int]]:
    robot_poses = load_robot_poses(args.record_file, args.euler_order)
    object_points = build_object_points((args.cols, args.rows), args.square_size)
    views: list[ViewData] = []
    image_size = None
    excluded = set(getattr(args, 'exclude_images', []))

    for image_path in numeric_image_files(args.image_dir):
        image_id = int(image_path.stem)
        if image_id in excluded:
            print(f"[skip] {image_path.name}: explicitly excluded by quality review")
            continue
        if image_id >= len(robot_poses):
            print(f"[skip] {image_path.name}: no pose line {image_id} in {args.record_file}")
            continue
        ok, current_size, corners = detect_chessboard(image_path, (args.cols, args.rows), args.use_sb_first)
        if current_size is not None:
            if image_size is not None and current_size != image_size:
                print(f"[skip] {image_path.name}: image size {current_size} differs from {image_size}")
                continue
            image_size = current_size
        if not ok or corners is None:
            print(f"[skip] {image_path.name}: chessboard not found")
            continue
        views.append(
            ViewData(
                image_id=image_id,
                image_path=image_path,
                image_points=corners.reshape(-1, 2).astype(np.float64),
                object_points=object_points.astype(np.float64),
                T_base_gripper=robot_poses[image_id],
            )
        )

    if image_size is None:
        raise RuntimeError(f"No readable images found in {args.image_dir}")
    if len(views) < 3:
        raise RuntimeError(f"Need at least 3 valid views, got {len(views)}")
    return views, image_size


def load_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None]:
    if args.intrinsics_json is None:
        return None, None
    data = json.loads(args.intrinsics_json.read_text(encoding="utf-8"))
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    else:
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(data.get("dist", data.get("dist_coeffs", [0, 0, 0, 0, 0])), dtype=np.float64).reshape(1, -1)
    return K, dist


def estimate_camera_target_poses(
    views: list[ViewData],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    K, dist = load_intrinsics(args)
    if args.intrinsics_json is not None:
        intrinsics = json.loads(args.intrinsics_json.read_text(encoding="utf-8"))
        if (intrinsics.get('width'), intrinsics.get('height')) != image_size:
            raise ValueError(f"内参分辨率 {(intrinsics.get('width'), intrinsics.get('height'))} 与图片分辨率 {image_size} 不符")
    object_points = [v.object_points.astype(np.float32) for v in views]
    image_points = [v.image_points.reshape(-1, 1, 2).astype(np.float32) for v in views]

    if K is None:
        print(
            "\n" + "!" * 78 + "\n"
            "[警告] 未提供相机内参，将用棋盘格图片自标定内参。\n"
            "       近似共面/视角有限的棋盘图自标定出的焦距、主点往往严重偏离真实值，\n"
            "       会被手眼外参吸收，导致标定结果在实际抓取中出现系统性偏差。\n"
            "       请用采集时相同相机及分辨率的 CameraInfo 或 RealSense 内参。\n"
            + "!" * 78 + "\n"
        )
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    else:
        print(f"[内参] 使用外部内参 (固定不优化): fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        rvecs = []
        tvecs = []
        all_residuals = []
        for view in views:
            ok, rvec, tvec = cv2.solvePnP(
                view.object_points,
                view.image_points,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise RuntimeError(f"solvePnP failed for {view.image_path}")
            projected, _ = cv2.projectPoints(view.object_points, rvec, tvec, K, dist)
            all_residuals.append((projected.reshape(-1, 2) - view.image_points).reshape(-1))
            rvecs.append(rvec)
            tvecs.append(tvec)
        rms = float(np.sqrt(np.mean(np.concatenate(all_residuals) ** 2)))

    for view, rvec, tvec in zip(views, rvecs, tvecs):
        R_camera_target, _ = cv2.Rodrigues(rvec)
        view.T_camera_target = make_transform(R_camera_target, tvec)
    return K, dist, float(rms)


def board_consistency_errors(T_base_camera: np.ndarray, views: list[ViewData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_gripper_targets = [
        invert_transform(view.T_base_gripper) @ T_base_camera @ view.T_camera_target
        for view in views
    ]
    T_mean = average_transform(T_gripper_targets)
    rot_errors = np.array([
        rotation_angle_deg(T_mean[:3, :3], T[:3, :3])
        for T in T_gripper_targets
    ])
    trans_errors = np.array([
        np.linalg.norm(T_mean[:3, 3] - T[:3, 3])
        for T in T_gripper_targets
    ])
    return rot_errors, trans_errors, T_mean


def reprojection_rms(
    views: list[ViewData],
    K: np.ndarray,
    dist: np.ndarray,
    T_base_camera: np.ndarray,
    T_gripper_target: np.ndarray,
) -> float:
    residuals = reprojection_residual_vector(views, K, dist, T_base_camera, T_gripper_target)
    return float(np.sqrt(np.mean(residuals ** 2)))


def reprojection_residual_vector(
    views: list[ViewData],
    K: np.ndarray,
    dist: np.ndarray,
    T_base_camera: np.ndarray,
    T_gripper_target: np.ndarray,
) -> np.ndarray:
    T_camera_base = invert_transform(T_base_camera)
    residuals = []
    for view in views:
        T_camera_target = T_camera_base @ view.T_base_gripper @ T_gripper_target
        rvec, _ = cv2.Rodrigues(T_camera_target[:3, :3])
        projected, _ = cv2.projectPoints(
            view.object_points,
            rvec,
            T_camera_target[:3, 3],
            K,
            dist,
        )
        residuals.append((projected.reshape(-1, 2) - view.image_points).reshape(-1))
    return np.concatenate(residuals)


def evaluate_result(
    name: str,
    T_base_camera: np.ndarray,
    views: list[ViewData],
    K: np.ndarray,
    dist: np.ndarray,
) -> CalibrationResult:
    rot_errors, trans_errors, T_gripper_target = board_consistency_errors(T_base_camera, views)
    return CalibrationResult(
        name=name,
        T_base_camera=T_base_camera,
        T_gripper_target=T_gripper_target,
        board_rot_mean_deg=float(np.mean(rot_errors)),
        board_rot_max_deg=float(np.max(rot_errors)),
        board_trans_mean_m=float(np.mean(trans_errors)),
        board_trans_max_m=float(np.max(trans_errors)),
        reproj_rms_px=reprojection_rms(views, K, dist, T_base_camera, T_gripper_target),
    )


def solve_closed_form(views: list[ViewData], K: np.ndarray, dist: np.ndarray) -> list[CalibrationResult]:
    # OpenCV calibrateHandEye expects T_base_gripper for eye-in-hand.
    # For eye-to-hand with the target on the gripper, pass T_gripper_base instead;
    # the returned "camera-to-gripper" transform is then T_base_camera.
    T_gripper_base = [invert_transform(view.T_base_gripper) for view in views]
    R_gripper2base = [T[:3, :3] for T in T_gripper_base]
    t_gripper2base = [T[:3, 3].reshape(3, 1) for T in T_gripper_base]
    R_target2cam = [view.T_camera_target[:3, :3] for view in views]
    t_target2cam = [view.T_camera_target[:3, 3].reshape(3, 1) for view in views]

    results = []
    for name, method in HAND_EYE_METHODS.items():
        try:
            R_base_camera, t_base_camera = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=method,
            )
        except cv2.error as exc:
            print(f"[skip] {name}: {exc}")
            continue
        if np.linalg.det(R_base_camera) < 0.5:
            print(f"[skip] {name}: invalid rotation determinant {np.linalg.det(R_base_camera):.6f}")
            continue
        T_base_camera = make_transform(R_base_camera, t_base_camera)
        results.append(evaluate_result(name, T_base_camera, views, K, dist))
    return results


def refine_with_reprojection(
    initial: CalibrationResult,
    views: list[ViewData],
    K: np.ndarray,
    dist: np.ndarray,
    max_nfev: int,
) -> CalibrationResult:
    r_bc, t_bc = transform_to_rt(initial.T_base_camera)
    r_gt, t_gt = transform_to_rt(initial.T_gripper_target)
    x0 = np.r_[r_bc, t_bc, r_gt, t_gt]

    def residual(x: np.ndarray) -> np.ndarray:
        T_base_camera = rt_to_transform(x[:3], x[3:6])
        T_gripper_target = rt_to_transform(x[6:9], x[9:12])
        return reprojection_residual_vector(views, K, dist, T_base_camera, T_gripper_target)

    result = least_squares(
        residual,
        x0,
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
    )
    T_base_camera = rt_to_transform(result.x[:3], result.x[3:6])
    refined = evaluate_result("REFINED_REPROJECTION", T_base_camera, views, K, dist)
    refined.T_gripper_target = rt_to_transform(result.x[6:9], result.x[9:12])
    refined.reproj_rms_px = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return refined


def result_score(result: CalibrationResult) -> float:
    # Keep rotation and translation on comparable human-readable scales.
    return result.board_rot_mean_deg + result.board_trans_mean_m * 100.0


def print_transform(title: str, T: np.ndarray) -> None:
    quat_xyzw = normalize_quat_sign(Rotation.from_matrix(T[:3, :3]).as_quat())
    print(f"\n{title}")
    print("translation [m]:")
    print(T[:3, 3])
    print("quaternion [qx, qy, qz, qw]:")
    print(quat_xyzw)
    print("rotation matrix:")
    print(T[:3, :3])


def print_result_table(results: list[CalibrationResult]) -> None:
    print("\n方法对比:")
    print("method                 rot_mean  rot_max   trans_mean  trans_max   reproj_rms")
    print("                       [deg]     [deg]     [mm]        [mm]        [px]")
    for r in results:
        print(
            f"{r.name:<22}"
            f"{r.board_rot_mean_deg:>8.4f}  "
            f"{r.board_rot_max_deg:>8.4f}  "
            f"{r.board_trans_mean_m * 1000.0:>10.3f}  "
            f"{r.board_trans_max_m * 1000.0:>9.3f}  "
            f"{r.reproj_rms_px:>9.4f}"
        )


def save_json(path: Path, final_result: CalibrationResult, K: np.ndarray, dist: np.ndarray,
              views: list[ViewData], cols: int, rows: int, square_size: float) -> None:
    T = final_result.T_base_camera
    quat = normalize_quat_sign(Rotation.from_matrix(T[:3, :3]).as_quat())
    data = {
        "convention": "P_base = R_base_camera @ P_camera + t_base_camera",
        "method": final_result.name,
        "used_image_ids": [view.image_id for view in views],
        "chessboard_cols": cols,
        "chessboard_rows": rows,
        "square_size_m": square_size,
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "T_base_camera": T.tolist(),
        "T_gripper_target": final_result.T_gripper_target.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quaternion_xyzw": quat.tolist(),
        "board_rot_mean_deg": final_result.board_rot_mean_deg,
        "board_rot_max_deg": final_result.board_rot_max_deg,
        "board_trans_mean_m": final_result.board_trans_mean_m,
        "board_trans_max_m": final_result.board_trans_max_m,
        "reproj_rms_px": final_result.reproj_rms_px,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baxter 固定相机眼在手外标定（只使用有效棋盘格图片）")
    parser.add_argument("--dataset", type=Path, help="record_baxter.py 输出目录，自动读取 picture/、record.txt 和 intrinsics.json")
    parser.add_argument("--image-dir", type=Path, help="手工指定图片目录")
    parser.add_argument("--record-file", type=Path, help="手工指定位姿记录")
    parser.add_argument("--cols", type=int, default=11, help="棋盘格列内角点数，默认 11")
    parser.add_argument("--rows", type=int, default=8, help="棋盘格行内角点数，默认 8")
    parser.add_argument("--square-size", type=float, default=0.015, help="chessboard square size in meters")
    parser.add_argument("--euler-order", default="xyz", help="robot pose Euler order used in record.txt")
    parser.add_argument("--intrinsics-json", type=Path, help="该相机的内参；配合 --dataset 自动读取")
    parser.add_argument("--output-json", type=Path, help="结果路径；配合 --dataset 保存到该数据目录")
    parser.add_argument("--no-refine", action="store_true", help="disable nonlinear reprojection refinement")
    parser.add_argument("--exclude-images", type=int, nargs='+', default=[], metavar='ID',
                        help="排除质检确认有问题的图片编号；原始图像和位姿不变，例如 --exclude-images 5 7")
    parser.add_argument(
        "--use-sb-first", action=argparse.BooleanOptionalAction, default=True,
        help="优先使用精度更高的 SB 棋盘格角点检测（默认开启；失败时回退普通检测）",
    )
    parser.add_argument("--max-nfev", type=int, default=2000)
    args = parser.parse_args()
    if args.dataset is not None:
        image_dir = args.dataset / 'picture'
        args.image_dir = args.image_dir or (image_dir if image_dir.is_dir() else args.dataset / 'rgb')
        args.record_file = args.record_file or args.dataset / 'record.txt'
        args.intrinsics_json = args.intrinsics_json or args.dataset / 'intrinsics.json'
        args.output_json = args.output_json or args.dataset / 'hand_eye_result.json'
    if args.image_dir is None or args.record_file is None or args.intrinsics_json is None:
        parser.error('请指定 --dataset，或提供 --image-dir、--record-file、--intrinsics-json')
    args.output_json = args.output_json or Path('hand_eye_result.json')
    if args.cols < 2 or args.rows < 2 or args.square_size <= 0:
        parser.error('棋盘格内角点数至少为 2，方格边长必须大于零')
    return args


def main() -> None:
    args = parse_args()
    views, image_size = load_views(args)
    K, dist, camera_rms = estimate_camera_target_poses(views, image_size, args)

    print(f"有效图片数量: {len(views)}")
    print("使用图片编号:", [view.image_id for view in views])
    print(f"图像尺寸: {image_size[0]}x{image_size[1]}")
    print(f"相机/PnP RMS: {camera_rms:.6f} px")
    print("camera_matrix:")
    print(K)
    print("dist_coeffs:")
    print(dist.reshape(-1))

    results = solve_closed_form(views, K, dist)
    if not results:
        raise RuntimeError("No valid hand-eye result was produced")

    results.sort(key=result_score)
    best_closed_form = results[0]

    if not args.no_refine:
        refined = refine_with_reprojection(best_closed_form, views, K, dist, args.max_nfev)
        results.append(refined)
        final_result = refined
    else:
        final_result = best_closed_form

    print_result_table(results)
    print_transform("推荐结果 T_base_camera / 相机坐标 -> 基座坐标", final_result.T_base_camera)
    print_transform("同时估计的 T_gripper_target / 标定板坐标 -> 末端坐标", final_result.T_gripper_target)
    if (final_result.board_rot_max_deg > 5.0 or final_result.board_trans_max_m > 0.03
            or final_result.reproj_rms_px > 5.0):
        print("\n[注意] 一致性或重投影误差较大，可能有误检、镜像角点或未停稳的样本；"
              "请检查有效图片和位姿，再用于实际定位。")

    save_json(args.output_json, final_result, K, dist, views, args.cols, args.rows, args.square_size)
    print(f"\n结果已保存: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
