#!/usr/bin/env python3
"""固定相机 + Baxter 末端位姿的眼在手外标定采集；窗口内 a 保存，q 退出。"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, EndpointState
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState


JOINT_SUFFIXES = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')
JOINT_LIMITS = {
    's0': (-1.70167993878, 1.70167993878), 's1': (-2.147, 1.047),
    'e0': (-3.05417993878, 3.05417993878), 'e1': (-0.05, 2.618),
    'w0': (-3.059, 3.059), 'w1': (-1.57079632679, 2.094),
    'w2': (-3.059, 3.059),
}


def arm_joint_names(arm):
    return tuple(f'{arm}_{suffix}' for suffix in JOINT_SUFFIXES)


def joint_values(msg, arm):
    states = dict(zip(msg.name, msg.position))
    names = arm_joint_names(arm)
    if any(name not in states for name in names):
        raise ValueError(f'/robot/joint_states 缺少 {arm} 臂的七个关节')
    values = {name: float(states[name]) for name in names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError('关节状态包含无效角度')
    return values


def validate_action(data, arm):
    if data.get('format') != 'baxter_joint_sample_v1' or data.get('arm') != arm:
        raise ValueError('动作文件不是当前机械臂的 Baxter 关节记录')
    positions = data['joint_positions_rad']
    if set(positions) != set(arm_joint_names(arm)):
        raise ValueError('动作文件必须包含当前手臂的七个关节，不能混用另一只手臂')
    result = {}
    for suffix in JOINT_SUFFIXES:
        name = f'{arm}_{suffix}'
        value = float(positions[name])
        lower, upper = JOINT_LIMITS[suffix]
        if not np.isfinite(value) or not lower - 0.01 <= value <= upper + 0.01:
            raise ValueError(f'{name}={value} 超出 Baxter URDF 关节范围 [{lower}, {upper}]')
        result[name] = value
    return result


def load_actions(path, arm):
    if path.is_dir() and (path / 'action').is_dir():
        path = path / 'action'
    if path.is_dir():
        files = sorted((p for p in path.glob('*.json') if p.stem.isdecimal()), key=lambda p: int(p.stem))
        if [int(p.stem) for p in files] != list(range(len(files))):
            raise ValueError(f'{path} 中的动作编号必须从 0 连续排列')
    else:
        files = [path]
    if not files:
        raise ValueError(f'{path} 中没有可用动作')
    return [(file, validate_action(json.loads(file.read_text(encoding='utf-8')), arm)) for file in files]


def image_to_bgr(msg):
    """直接解码 ROS Image，避免 cv_bridge 与 conda OpenCV 的 ABI 冲突。"""
    conversions = {
        'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR),
        'mono8': (1, cv2.COLOR_GRAY2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
        'rgba8': (4, cv2.COLOR_RGBA2BGR),
    }
    if msg.encoding not in conversions:
        raise ValueError(f'不支持的彩色图像编码 {msg.encoding!r}')
    channels, conversion = conversions[msg.encoding]
    if msg.step < msg.width * channels or len(msg.data) < msg.step * msg.height:
        raise ValueError('ROS Image 的 step/data 长度不正确')
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = raw[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        pixels = pixels[:, :, 0]
    return cv2.cvtColor(pixels, conversion) if conversion is not None else pixels.copy()


def pose_values(msg):
    p, q = msg.pose.position, msg.pose.orientation
    xyz = np.array([p.x, p.y, p.z], dtype=float)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise ValueError('末端位姿中存在非有限数值')
    if abs(np.linalg.norm(quat) - 1) > 0.01:
        raise ValueError('末端位姿四元数无效')
    return xyz, Rotation.from_quat(quat)


def stationary_pose(before, after):
    xyz0, rot0 = pose_values(before)
    xyz1, rot1 = pose_values(after)
    if np.linalg.norm(xyz1 - xyz0) > 0.002 or (rot0.inv() * rot1).magnitude() > np.deg2rad(0.5):
        raise ValueError('前后位姿差超过 2 mm / 0.5°；停稳后重试')
    return [*xyz0.tolist(), *rot0.as_euler('xyz', degrees=False).tolist()]


def check_intrinsics(data):
    width, height = int(data['width']), int(data['height'])
    K = np.asarray(data['camera_matrix'], dtype=float).reshape(3, 3)
    dist = np.asarray(data.get('dist_coeffs', data.get('dist', [])), dtype=float).reshape(-1)
    if dist.size == 0:
        dist = np.zeros(5, dtype=float)
        data['dist_coeffs'] = dist.tolist()
    if width <= 0 or height <= 0 or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('相机内参分辨率或焦距无效')
    if not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError('相机内参含非有限数值')
    if dist.size not in (4, 5, 8, 12, 14):
        raise ValueError(f'OpenCV 不支持 {dist.size} 个畸变系数')
    if 'inverse_brown_conrady' in data.get('model', '').lower() and np.any(abs(dist) > 1e-8):
        raise ValueError('非零 inverse_brown_conrady 畸变无法直接用于 OpenCV PnP；请使用已矫正的彩色图及对应内参')
    return data


def info_intrinsics(msg):
    if msg.distortion_model not in ('plumb_bob', 'rational_polynomial', 'none', ''):
        raise ValueError(f'不支持的 CameraInfo 畸变模型：{msg.distortion_model}')
    K = np.array(msg.K, dtype=float).reshape(3, 3)
    return check_intrinsics({
        'width': msg.width, 'height': msg.height, 'model': msg.distortion_model,
        'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2],
        'camera_matrix': K.tolist(), 'dist_coeffs': list(msg.D),
    })


class Receiver:
    def __init__(self, arm):
        self.condition = threading.Condition()
        self.state = self.frame = None
        self.arm = arm
        self.joint_state = None
        self.state_seq = self.frame_seq = self.joint_seq = 0
        self.state_time = self.frame_time = self.joint_time = 0.0
        self.state_sub = rospy.Subscriber(
            f'/robot/limb/{arm}/endpoint_state', EndpointState, self.on_state, queue_size=1, tcp_nodelay=True
        )
        self.joint_sub = rospy.Subscriber('/robot/joint_states', JointState, self.on_joint, queue_size=1,
                                          tcp_nodelay=True)
        self.image_sub = None

    def on_state(self, msg):
        with self.condition:
            self.state, self.state_time = msg, time.monotonic()
            self.state_seq += 1
            self.condition.notify_all()

    def on_joint(self, msg):
        # 同一话题也可能有夹爪状态发布者；只接受完整的七轴机械臂状态。
        try:
            joint_values(msg, self.arm)
        except ValueError:
            return
        with self.condition:
            self.joint_state, self.joint_time = msg, time.monotonic()
            self.joint_seq += 1
            self.condition.notify_all()

    def on_image(self, msg):
        try:
            frame = image_to_bgr(msg)
        except Exception as exc:
            rospy.logwarn_throttle(5, '无法解码彩色图像：%s', exc)
            return
        self.set_frame(frame, msg.header.stamp.to_sec(), msg.header.frame_id)

    def set_frame(self, frame, stamp, frame_id):
        with self.condition:
            self.frame, self.frame_time = frame, time.monotonic()
            self.image_stamp, self.image_frame_id = stamp, frame_id
            self.frame_seq += 1
            self.condition.notify_all()

    def subscribe_image(self, topic):
        self.image_sub = rospy.Subscriber(topic, Image, self.on_image, queue_size=1, buff_size=2**24)

    def wait_for(self, predicate, seconds, error):
        deadline = time.monotonic() + seconds
        with self.condition:
            while not predicate():
                if rospy.is_shutdown() or time.monotonic() >= deadline:
                    raise RuntimeError(error)
                self.condition.wait(min(0.1, deadline - time.monotonic()))

    def capture(self, intrinsics):
        # 消费两次更新的末端和关节状态，将一帧新图夹在它们之间。
        with self.condition:
            prior_state, prior_joint = self.state_seq, self.joint_seq
            self.wait_for(lambda: self.state_seq > prior_state and self.joint_seq > prior_joint, 4,
                          '收不到新的末端位姿或关节状态；检查 Baxter 的 ROS 网络')
            before, before_time = self.state, self.state_time
            joint_before, joint_before_time = self.joint_state, self.joint_time
            prior_frame = self.frame_seq
            self.wait_for(lambda: self.frame_seq > prior_frame, 4, '收不到新的彩色相机图像')
            frame, frame_time = self.frame.copy(), self.frame_time
            stamp, frame_id = self.image_stamp, self.image_frame_id
            state_at_frame, joint_at_frame = self.state_seq, self.joint_seq
            self.wait_for(lambda: self.state_seq > state_at_frame and self.joint_seq > joint_at_frame, 4,
                          '图像之后未收到末端位姿或关节状态')
            after, after_time = self.state, self.state_time
            joint_after, joint_after_time = self.joint_state, self.joint_time
        if (not before_time <= frame_time <= after_time or after_time - before_time > 1.0
                or not joint_before_time <= frame_time <= joint_after_time
                or joint_after_time - joint_before_time > 1.0):
            raise ValueError('图像与末端/关节状态本机接收时间间隔超过 1 秒')
        if frame.shape[:2] != (intrinsics['height'], intrinsics['width']):
            raise ValueError(f'图像 {frame.shape[1]}x{frame.shape[0]} 与内参分辨率不一致')
        if before.header.frame_id != after.header.frame_id:
            raise ValueError('Baxter 末端坐标系标识发生变化')
        pose = stationary_pose(before, after)
        before_joints, after_joints = joint_values(joint_before, self.arm), joint_values(joint_after, self.arm)
        if max(abs(after_joints[name] - value) for name, value in before_joints.items()) > 0.01:
            raise ValueError('拍照前后关节变化超过 0.01 rad，停稳后再拍')
        return frame, pose, before_joints, {
            'image_header_stamp': stamp, 'image_frame_id': frame_id,
            'endpoint_frame_id': before.header.frame_id,
            'endpoint_frame_note': 'Baxter SDK endpoint pose uses its base; ROS header.frame_id may be empty',
            'state_before_monotonic': before_time, 'image_received_monotonic': frame_time,
            'state_after_monotonic': after_time, 'pose_xyz_rpy': pose,
            'joint_before_monotonic': joint_before_time, 'joint_after_monotonic': joint_after_time,
            'pairing': 'stationary capture, ordered local receipt times; not hardware synchronization',
        }

    def close(self):
        if self.image_sub is not None:
            self.image_sub.unregister()
        self.state_sub.unregister()
        self.joint_sub.unregister()


class RealSenseSource:
    def __init__(self, receiver, args):
        import pyrealsense2 as rs
        self.rs, self.receiver = rs, receiver
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if args.serial:
            cfg.enable_device(args.serial)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.profile = self.pipeline.start(cfg)
        self.stop_event = threading.Event()
        self.thread = None
        try:
            active_device = self.profile.get_device()
            self.serial = active_device.get_info(rs.camera_info.serial_number)
            intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
            self.map_x = self.map_y = None
            if 'inverse_brown_conrady' in str(intr.model) and any(abs(k) > 1e-8 for k in intr.coeffs):
                self.map_x, self.map_y = cv2.initUndistortRectifyMap(
                    np.asarray(K, dtype=float), np.asarray(intr.coeffs, dtype=float),
                    np.eye(3), np.asarray(K, dtype=float), (intr.width, intr.height), cv2.CV_32FC1
                )
            self.intrinsics = check_intrinsics({
                'width': intr.width, 'height': intr.height,
                'model': 'plumb_bob' if self.map_x is not None else str(intr.model),
                'fx': intr.fx, 'fy': intr.fy, 'cx': intr.ppx, 'cy': intr.ppy,
                'camera_matrix': K, 'dist_coeffs': [0.0] * 5 if self.map_x is not None else list(intr.coeffs),
                'serial': self.serial, 'rectified': self.map_x is not None,
            })
        except Exception:
            self.pipeline.stop()
            raise

    def start(self):
        def read_frames():
            while not self.stop_event.is_set() and not rospy.is_shutdown():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color = frames.get_color_frame()
                    if color:
                        image = np.asanyarray(color.get_data()).copy()
                        if self.map_x is not None:
                            image = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
                        self.receiver.set_frame(image,
                                                color.get_timestamp() / 1000, f'realsense/{self.serial}/color')
                except RuntimeError as exc:
                    if not self.stop_event.is_set():
                        rospy.logwarn_throttle(5, 'RealSense 获取图像失败：%s', exc)
        self.thread = threading.Thread(target=read_frames, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.pipeline.stop()
        if self.thread is not None:
            self.thread.join(timeout=2)


class Dataset:
    def __init__(self, folder, intrinsics, arm, source, action_source=None):
        self.folder, self.count, self.arm = folder, 0, arm
        self.lines = []
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'picture').mkdir()
        (folder / 'action').mkdir()
        (folder / 'intrinsics.json').write_text(json.dumps(intrinsics, indent=2, ensure_ascii=False) + '\n')
        (folder / 'session.json').write_text(json.dumps({
            'robot': 'Baxter', 'arm': arm, 'image_source': source,
            'replayed_from': str(action_source.resolve()) if action_source else None,
            'endpoint_topic': f'/robot/limb/{arm}/endpoint_state',
            'joint_topic': '/robot/joint_states',
            'pose_frame': 'Baxter base; confirm before using the transform',
            'pose_unit': 'm', 'angles_unit': 'rad', 'euler_order': 'xyz',
        }, indent=2, ensure_ascii=False) + '\n')
        (folder / 'record.txt').write_text('')

    def save(self, frame, pose, joints, metadata):
        img = self.folder / 'picture' / f'{self.count}.png'
        sample = self.folder / 'action' / f'{self.count}.json'
        temp = self.folder / 'record.tmp'
        line = ','.join(f'{v:.12g}' for v in pose) + '\n'
        action = {
            'format': 'baxter_joint_sample_v1', 'arm': self.arm,
            'image': f'{self.count}.png', 'joint_positions_rad': joints,
            'endpoint_pose_xyz_rpy': pose, **metadata,
        }
        try:
            if not cv2.imwrite(str(img), frame):
                raise RuntimeError(f'图片保存失败: {img}')
            sample.write_text(json.dumps(action, indent=2, ensure_ascii=False) + '\n')
            temp.write_text(''.join(self.lines) + line)
            temp.replace(self.folder / 'record.txt')
        except Exception:
            img.unlink(missing_ok=True)
            sample.unlink(missing_ok=True)
            temp.unlink(missing_ok=True)
            raise
        self.lines.append(line)
        self.count += 1
        print(f'已保存第 {self.count} 组：{img}', flush=True)


def next_output_dir(root):
    for index in range(1, 1_000_000):
        folder = root / str(index)
        if not folder.exists():
            return folder
    raise RuntimeError('编号目录过多，请显式传入 --output')


def replay(actions, limb, receiver, dataset, intrinsics, args, stop_event):
    """仅当实际关节角到位且停稳，才写入新的一组图像/动作。"""
    limb.set_joint_position_speed(args.speed)
    try:
        for source, target in actions:
            if stop_event.is_set():
                break
            print(f'正在运动到 {source.name}，最多 {args.move_timeout:g} 秒；随后停稳 {args.settle_seconds:g} 秒', flush=True)
            limb.move_to_joint_positions(target, timeout=args.move_timeout, threshold=0.01,
                                         test=stop_event.is_set)
            if stop_event.is_set():
                limb.set_joint_positions(limb.joint_angles())
                break
            with receiver.condition:
                old = receiver.joint_seq
                receiver.wait_for(lambda: receiver.joint_seq > old, 4, '无法验证机器人实际关节角')
                actual = joint_values(receiver.joint_state, args.arm)
            error = max(abs(actual[name] - value) for name, value in target.items())
            if error > 0.02:
                limb.set_joint_positions(limb.joint_angles())
                raise RuntimeError(f'{source.name} 未到位（最大关节误差 {error:.4f} rad），停止回放，不拍照')
            if stop_event.wait(args.settle_seconds):
                limb.set_joint_positions(limb.joint_angles())
                break
            frame, pose, joints, metadata = receiver.capture(intrinsics)
            error = max(abs(joints[name] - value) for name, value in target.items())
            if error > 0.02:
                raise RuntimeError(f'{source.name} 拍照时已偏离目标 {error:.4f} rad，不保存该照片')
            if stop_event.is_set():
                break
            metadata['replayed_from'] = str(source.resolve())
            dataset.save(frame, pose, joints, metadata)
    finally:
        limb.set_joint_position_speed(0.3)  # 恢复 SDK 默认速度比例。


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('right', 'left'), default='right')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image-topic', help='固定相机的 ROS 彩色 Image 话题')
    source.add_argument('--realsense', action='store_true', help='本机 USB RealSense 固定相机')
    source.add_argument('--list', action='store_true', help='列出当前 ROS 图像和内参话题后退出')
    params = parser.add_mutually_exclusive_group()
    params.add_argument('--camera-info-topic', help='该图像对应的 sensor_msgs/CameraInfo 话题')
    params.add_argument('--intrinsics', type=Path, help='该相机当前分辨率的内参 JSON')
    parser.add_argument('--serial', help='本地 RealSense 相机序列号，仅 --realsense')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--output', type=Path, help='新建输出目录；默认依次选择 1、2、3……')
    parser.add_argument('--action', type=Path, help='回放一个 action/N.json 或整个 action 目录，自动采图')
    parser.add_argument('--move-timeout', type=float, default=10.0, help='每个关节动作到位超时秒数，默认 10')
    parser.add_argument('--settle-seconds', type=float, default=10.0, help='确认到位后等待停稳秒数，默认 10')
    parser.add_argument('--speed', type=float, default=0.15, help='回放时 SDK 最大关节速度比例，默认 0.15')
    args = parser.parse_args(rospy.myargv()[1:])
    if not (args.list or args.realsense or args.image_topic):
        parser.error('请指定 --image-topic、--realsense 或 --list')
    if args.image_topic and not (args.camera_info_topic or args.intrinsics):
        parser.error('ROS 相机必须指定 --camera-info-topic 或 --intrinsics')
    if args.realsense and (args.camera_info_topic or args.intrinsics):
        parser.error('--realsense 会直接读取对应相机内参，不需另指定内参')
    if args.serial and not args.realsense:
        parser.error('--serial 只能与 --realsense 配合使用')
    if min(args.width, args.height, args.fps) <= 0:
        parser.error('图像宽高、帧率必须为正数')
    if args.move_timeout <= 0 or args.settle_seconds < 0 or not 0 < args.speed <= 1:
        parser.error('--move-timeout 要大于 0、--settle-seconds 不能为负、--speed 必须在 (0,1]')
    if args.list and args.action:
        parser.error('--list 不接受 --action')
    return args


def main():
    args = options()
    actions = load_actions(args.action, args.arm) if args.action else None
    rospy.init_node('baxter_eye_to_hand_capture', anonymous=True)
    if args.list:
        for name, kind in rospy.get_published_topics():
            if kind in ('sensor_msgs/Image', 'sensor_msgs/CameraInfo'):
                print(name, kind)
        return
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise SystemExit('请在有图形桌面的本机终端启动，或配置 X11 转发')

    topics = dict(rospy.get_published_topics())
    if args.image_topic:
        if topics.get(args.image_topic) != 'sensor_msgs/Image':
            raise SystemExit(f'未找到 sensor_msgs/Image 彩色话题: {args.image_topic}')
        if any(part.endswith('_hand_camera') for part in args.image_topic.strip('/').split('/')):
            raise SystemExit('手腕相机会随手臂运动，不能直接作为固定相机进行眼在手外标定；请指定固定相机')
        if args.camera_info_topic:
            if topics.get(args.camera_info_topic) != 'sensor_msgs/CameraInfo':
                raise SystemExit(f'未找到 sensor_msgs/CameraInfo: {args.camera_info_topic}')
            print(f'正在等待相机内参：{args.camera_info_topic}（最多 30 秒）……', flush=True)
            try:
                info = rospy.wait_for_message(args.camera_info_topic, CameraInfo, timeout=30)
            except rospy.ROSException as exc:
                raise RuntimeError('收不到相机内参；先在 camera 目录执行 '
                                   'bash camera_service.sh restart --fps 15，确认发布就绪后重新采集。'
                                   '独立 camera_demo.py 预览不会发布内参。') from exc
            intrinsics = info_intrinsics(info)
        else:
            intrinsics = check_intrinsics(json.loads(args.intrinsics.read_text()))
    receiver = Receiver(args.arm)
    camera = None
    try:
        if args.realsense:
            camera = RealSenseSource(receiver, args)
            intrinsics = camera.intrinsics
            camera.start()
            name = f'本机 RealSense 彩色流（序列号 {camera.serial}）'
        else:
            receiver.subscribe_image(args.image_topic)
            name = args.image_topic
        print('正在等待 Baxter 首次末端位姿和关节状态（ROS 首次连接可能需要几十秒）……', flush=True)
        receiver.wait_for(lambda: receiver.state_seq > 0 and receiver.joint_seq > 0, 45,
                          '45 秒内仍收不到 Baxter 末端或关节状态；请检查 rostopic hz 和机器人状态')
        receiver.wait_for(lambda: receiver.frame_seq > 0, 8, '收不到相机图像，请检查相机发布与 ROS 网络')
        if actions:
            print('正在检查机器人是否已使能（仅在 --action 回放模式下）……', flush=True)
            robot = rospy.wait_for_message('/robot/state', AssemblyState, timeout=35)
            if not robot.enabled or robot.stopped or robot.error:
                raise ValueError('机器人未使能、已停止或处于错误状态，不会回放动作')
            from baxter_interface import Limb
            limb = Limb(args.arm)
            print('正在等待 Baxter 运动命令接口连接（不会发送运动指令）……', flush=True)
            receiver.wait_for(lambda: limb._pub_joint_cmd.get_num_connections() > 0
                              and limb._pub_speed_ratio.get_num_connections() > 0, 45,
                              '45 秒内 Baxter 运动命令接口未连接；不会执行动作')
        folder = args.output or next_output_dir(Path(__file__).resolve().parent)
        dataset = Dataset(folder, intrinsics, args.arm, name, args.action)
        print(f'图像：{name}\n位姿：/robot/limb/{args.arm}/endpoint_state\n输出：{folder.resolve()}', flush=True)
        print('保持相机及 Baxter 底座不动；标定板装在所选手臂末端。', flush=True)
        if actions:
            print(f'自动回放 {len(actions)} 个动作，逐个到位、等待 {args.settle_seconds:g} 秒后拍照；按 q 停止。',
                  flush=True)
        else:
            print('手动移动手臂，停稳后按 a 同时保存照片、关节角与末端位姿；按 q 退出。', flush=True)
        results = queue.Queue()
        busy = False
        stop_event = threading.Event()
        worker_thread = None
        replay_error = None

        def capture_worker():
            try:
                results.put((receiver.capture(intrinsics), None))
            except Exception as exc:
                results.put((None, str(exc)))

        def playback_worker():
            try:
                replay(actions, limb, receiver, dataset, intrinsics, args, stop_event)
                results.put(('done', None))
            except Exception as exc:
                results.put(('error', str(exc)))

        window = 'Baxter eye-to-hand capture'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            if actions:
                worker_thread = threading.Thread(target=playback_worker, daemon=True)
                worker_thread.start()
            while not rospy.is_shutdown():
                try:
                    result, error = results.get_nowait()
                    if actions:
                        if result == 'error':
                            print('动作回放停止：' + error, flush=True)
                            replay_error = error
                        else:
                            print('所有动作已完成并自动拍照。', flush=True)
                        break
                    busy = False
                    if error:
                        print('本次未保存：' + error, flush=True)
                    else:
                        dataset.save(*result)
                except queue.Empty:
                    pass
                with receiver.condition:
                    frame = None if receiver.frame is None else receiver.frame.copy()
                    age = time.monotonic() - receiver.frame_time
                preview = np.zeros((480, 640, 3), np.uint8) if frame is None else frame
                label = 'Waiting for image' if frame is None else f'age={age:.1f}s'
                label += f' saved={dataset.count} ' + ('q: stop replay' if actions else
                                                  ('CAPTURING' if busy else 'a: save / q: quit'))
                cv2.putText(preview, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window, preview)
                key = cv2.waitKey(20) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('a') and not actions and not busy:
                    if frame is None or age > 1:
                        print('图像已过期；检查相机和 ROS 网络', flush=True)
                    else:
                        busy = True
                        threading.Thread(target=capture_worker, daemon=True).start()
        finally:
            stop_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=10)
        print(f'共保存 {dataset.count} 组，目录：{folder.resolve()}', flush=True)
        if replay_error:
            raise RuntimeError(replay_error)
    finally:
        if camera is not None:
            camera.close()
        receiver.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    except (RuntimeError, ValueError, rospy.ROSException) as exc:
        raise SystemExit(f'采集未完成：{exc}') from exc
