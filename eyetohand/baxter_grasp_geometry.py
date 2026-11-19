"""点选抓取的深度反投影、TCP 换算与路径计算；不连接机器人。"""
import numpy as np


def validate_transform(value):
    T = np.asarray(value, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
        raise ValueError('标定结果的 T_base_camera 不是有效的刚体变换')
    return T


def validate_intrinsics(intrinsics):
    K = np.asarray(intrinsics['camera_matrix'], dtype=float)
    dist = np.asarray(intrinsics['dist_coeffs'], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-8)
            or not np.isclose(K[2, 2], 1)
            or min(int(intrinsics['width']), int(intrinsics['height'])) <= 0):
        raise ValueError('彩色内参不是有效的针孔相机模型')
    if dist.size != 5 or not np.isfinite(dist).all() or np.any(np.abs(dist) > 1e-8):
        raise ValueError('抓取使用已矫正彩色图，需要五个全零畸变系数')
    return K


def validate_camera_info(msg, intrinsics):
    """核对流配置；仅允许焦距相对变化 0.1%，并返回实际使用的内参。"""
    K = validate_intrinsics(intrinsics)
    if (msg.width, msg.height) != (intrinsics['width'], intrinsics['height']):
        raise ValueError('当前相机分辨率与标定时不同')
    actual = np.asarray(msg.K, dtype=float).reshape(3, 3)
    if (msg.distortion_model != 'plumb_bob' or len(msg.D) != 5
            or not np.isfinite(msg.D).all() or np.any(np.abs(msg.D) > 1e-8)):
        raise ValueError('请订阅本机发布器矫正后的彩色图及对应内参')
    live = dict(width=msg.width, height=msg.height, model=msg.distortion_model,
                camera_matrix=actual.tolist(), dist_coeffs=list(msg.D))
    validate_intrinsics(live)
    focal_indices = ([0, 1], [0, 1])
    remaining = np.ones((3, 3), dtype=bool)
    remaining[focal_indices] = False
    if (not np.allclose(actual[focal_indices], K[focal_indices], atol=0, rtol=.001)
            or not np.allclose(actual[remaining], K[remaining], atol=.01, rtol=0)):
        raise ValueError('当前相机内参与标定时不同：'
                         f'当前 fx/fy/cx/cy={actual[0,0]:.3f}/{actual[1,1]:.3f}/'
                         f'{actual[0,2]:.3f}/{actual[1,2]:.3f}，'
                         f'标定={K[0,0]:.3f}/{K[1,1]:.3f}/{K[0,2]:.3f}/{K[1,2]:.3f}；'
                         '焦距容差为0.1%，主点容差为0.01像素，请选择对应的标定文件')
    return live


def depth_image(msg):
    """解码本机发布器的 16UC1 毫米深度；保留行填充和字节序。"""
    if msg.encoding != '16UC1':
        raise ValueError('对齐深度必须是 16UC1（毫米）；不能将米制浮点图当成毫米')
    if msg.step < msg.width * 2 or len(msg.data) != msg.step * msg.height:
        raise ValueError('深度图的 step/data 长度不正确')
    dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.step, 2)).astype(np.uint16, copy=True)


def point_from_depth(u, v, depth, intrinsics, transform):
    K = validate_intrinsics(intrinsics)
    if depth.dtype != np.uint16 or depth.shape != (intrinsics['height'], intrinsics['width']):
        raise ValueError('对齐深度的尺寸或格式与彩色图不一致')
    if not 2 <= u < depth.shape[1] - 2 or not 2 <= v < depth.shape[0] - 2:
        raise ValueError('请在图像内部选点')
    if depth[v, u] == 0:
        raise ValueError('点击处深度为零，请换一个有有效深度的点')
    patch = depth[v-2:v+3, u-2:u+3]
    valid = patch[patch > 0].astype(float) / 1000
    if valid.size < 13:
        raise ValueError('选点周围有效深度太少，请换点')
    if np.ptp(valid) > .02 + 1e-9:
        raise ValueError('选点周围深度跳变超过 20 mm，请避开物体边缘')
    z = float(np.median(valid))
    if not .15 <= z <= 2:
        raise ValueError('选点深度超出 0.15–2 m 的工作范围')
    camera = np.array([(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z])
    return camera, transform[:3, :3] @ camera + transform[:3, 3]


def grip_from_reported(reported, rotation, offset):
    return np.asarray(reported, dtype=float) + rotation.apply(offset)


def reported_for_grip(grip, rotation, offset):
    return np.asarray(grip, dtype=float) - rotation.apply(offset)


def target_plan(selection, reported, rotation, offset, clearance, inset, stop_above_m=0):
    """目标按基座 Z 抬高；保持按 p 时的姿态，并从修正后的目标退回。"""
    if not np.isfinite(stop_above_m) or stop_above_m < 0:
        raise ValueError('目标抬高距离必须是非负有限数值')
    start = grip_from_reported(reported, rotation, offset)
    # 保留相机测得的原始 surface，防止选点预览和按 p 重算时重复加偏移。
    surface = np.asarray(selection['surface'], dtype=float) + [0, 0, stop_above_m]
    vector = start - surface
    distance = np.linalg.norm(vector)
    if not np.isfinite(distance) or distance < 1e-6:
        raise ValueError('当前夹爪点与选中点重合，无法确定退回方向')
    outward = vector / distance
    grasp = surface - outward * inset
    retreat = surface + outward * min(clearance, distance)
    return dict(selection, start=start.tolist(), start_reported=np.asarray(reported).tolist(),
                adjusted_surface=surface.tolist(), stop_above_base_z_m=float(stop_above_m),
                grasp=grasp.tolist(), grasp_reported=reported_for_grip(grasp, rotation, offset).tolist(),
                retreat=retreat.tolist(), retreat_reported=reported_for_grip(retreat, rotation, offset).tolist(),
                quat=rotation.as_quat().tolist(), rpy_xyz_rad=rotation.as_euler('xyz').tolist(),
                tcp_offset_in_endpoint_m=np.asarray(offset).tolist())


def cartesian_points(start, target, step):
    start, target = np.asarray(start, float), np.asarray(target, float)
    if not np.isfinite(start).all() or not np.isfinite(target).all() or not np.isfinite(step) or step <= 0:
        raise ValueError('路径参数无效')
    count = max(1, int(np.ceil(np.linalg.norm(target-start) / step)))
    return [start + (target-start) * (i/count) for i in range(1, count+1)]
