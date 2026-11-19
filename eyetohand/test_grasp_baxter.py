#!/usr/bin/env python3
"""Baxter 右臂点选抓取；默认预览，--execute 启用按键动作。"""
import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
if Path(FONT_DIR).is_dir():
    os.environ.setdefault('QT_QPA_FONTDIR', FONT_DIR)
import cv2
if Path(FONT_DIR).is_dir():
    os.environ['QT_QPA_FONTDIR'] = FONT_DIR
import message_filters
import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, EndEffectorState, EndEffectorCommand, EndEffectorProperties, EndpointState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState

from record_baxter import image_to_bgr, joint_values, pose_values, validate_action
from baxter_grasp_geometry import (cartesian_points, depth_image, point_from_depth,
                                    target_plan, validate_camera_info, validate_intrinsics,
                                    validate_transform)


class Camera:
    """同步本机发布器同时间戳的彩色、对齐深度和 CameraInfo。"""
    def __init__(self, namespace, intrinsics):
        self.intrinsics = intrinsics
        self.lock = threading.Lock()
        self.bundle = None
        self.subs = [message_filters.Subscriber(namespace + suffix, kind, queue_size=1, buff_size=2**23)
                     for suffix, kind in [('/color/image_raw', Image),
                                          ('/aligned_depth_to_color/image_raw', Image),
                                          ('/color/camera_info', CameraInfo)]]
        self.sync = message_filters.TimeSynchronizer(self.subs, 8)
        self.sync.registerCallback(self.on_pair)

    def on_pair(self, color_msg, depth_msg, info):
        received = time.monotonic()
        try:
            live_intrinsics = validate_camera_info(info, self.intrinsics)
            headers = [m.header for m in (color_msg, depth_msg, info)]
            if (not headers[0].frame_id or any(h.frame_id != headers[0].frame_id for h in headers)
                    or not headers[0].stamp.to_sec()
                    or any(h.stamp != headers[0].stamp for h in headers)):
                raise ValueError('彩色、对齐深度、内参的坐标系或时间戳不一致')
            color, depth = image_to_bgr(color_msg), depth_image(depth_msg)
            if color.shape[:2] != depth.shape or depth.shape != (info.height, info.width):
                raise ValueError('彩色、深度及内参的尺寸不同')
            if (not getattr(self, '_reported_focal_difference', False)
                    and not np.allclose(live_intrinsics['camera_matrix'], self.intrinsics['camera_matrix'],
                                        atol=.01, rtol=0)):
                print('当前焦距与标定差异在0.1%以内，已接受；选点使用当前 CameraInfo 内参。', flush=True)
                self._reported_focal_difference = True
            with self.lock:
                self.bundle = (color, depth, received, headers[0].stamp.to_sec(), headers[0].frame_id,
                               live_intrinsics)
        except Exception as exc:
            with self.lock:
                self.bundle = None
            rospy.logwarn_throttle(5, '相机帧被拒绝：%s', exc)

    def snapshot(self):
        with self.lock:
            return self.bundle


class RightArm:
    def __init__(self, execute, speed, offset, step=.04, timeout=20):
        self.execute, self.speed = execute, speed
        self.offset = np.asarray(offset, float)
        self.step, self.timeout = step, timeout
        self.cancel = threading.Event()
        self.moving = threading.Event()
        self.lock = threading.RLock()
        self.condition = threading.Condition()
        self.endpoint = self.joints = self.robot = self.grip = self.grip_properties = None
        self.times = {}
        self.sequence = 0
        self.limb = None
        self.ik_service = None
        self.ik_lock = threading.Lock()
        self.ik_name = '/ExternalTools/right/PositionKinematicsNode/IKService'
        self.subs = [rospy.Subscriber(topic, kind, self.on_state, callback_args=key, queue_size=1)
                     for topic, kind, key in [('/robot/limb/right/endpoint_state', EndpointState, 'endpoint'),
                                              ('/robot/joint_states', JointState, 'joints'),
                                              ('/robot/state', AssemblyState, 'robot'),
                                              ('/robot/end_effector/right_gripper/state', EndEffectorState, 'grip'),
                                              ('/robot/end_effector/right_gripper/properties', EndEffectorProperties, 'grip_properties')]]
        # Gripper SDK construction configures hardware; use its documented command topic on key presses only.
        self.grip_pub = (rospy.Publisher('/robot/end_effector/right_gripper/command', EndEffectorCommand,
                                         queue_size=1) if execute else None)

    def on_state(self, msg, key):
        if key == 'joints':
            try:
                joint_values(msg, 'right')
            except ValueError:
                return
        with self.condition:
            setattr(self, key, msg)
            self.times[key] = time.monotonic()
            self.condition.notify_all()

    def wait(self, predicate, timeout, error):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not predicate():
                self.check_cancel()
                left = deadline-time.monotonic()
                if left <= 0:
                    raise RuntimeError(error)
                self.condition.wait(min(left, .05))

    def check_cancel(self):
        if self.cancel.is_set() or rospy.is_shutdown():
            raise RuntimeError('操作已取消')

    def require_fresh(self, keys):
        with self.condition:
            if any(time.monotonic()-self.times.get(k, -float('inf')) > 1 for k in keys):
                raise RuntimeError('机器人状态已过期，请检查 ROS 网络')

    def current(self):
        self.wait(lambda: self.endpoint is not None and self.joints is not None, 45,
                  '收不到右臂末端/七个关节状态；检查 ROS_MASTER_URI 和 ROS_IP')
        with self.condition:
            self.require_fresh(('endpoint', 'joints'))
            return pose_values(self.endpoint)

    def current_joints(self):
        with self.condition:
            self.require_fresh(('joints',))
            return joint_values(self.joints, 'right')

    def require_enabled(self):
        self.check_cancel()
        with self.condition:
            self.require_fresh(('endpoint', 'joints', 'robot'))
            if not self.robot.enabled or self.robot.stopped or self.robot.error or self.robot.estop_button:
                raise RuntimeError('Baxter 未使能、已停止或存在错误；请先检查机器人状态')

    def ensure_limb(self):
        if not self.execute:
            raise RuntimeError('当前仅预览，使用 --execute 启动后才允许动作')
        self.current()
        self.wait(lambda: self.robot is not None, 45, '收不到 /robot/state')
        self.require_enabled()
        if self.limb is None:
            import baxter_interface
            self.limb = baxter_interface.Limb('right')
        self.wait(lambda: self.limb._pub_joint_cmd.get_num_connections() > 0
                  and self.limb._pub_speed_ratio.get_num_connections() > 0, 5,
                  'Baxter 右臂运动命令接口未连接')
        self.check_cancel()

    def call_ik(self, point, quat, seed, timeout=15):
        self.check_cancel()
        request = SolvePositionIKRequest()
        pose = PoseStamped()
        pose.header.frame_id = 'base'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, point)
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, quat)
        request.pose_stamp = [pose]
        request.seed_angles = [JointState(name=list(seed), position=list(seed.values()))]
        request.seed_mode = request.SEED_USER
        answers = queue.Queue(maxsize=1)

        def request_service():
            # 机器人 .local 主机名首次解析约需 10 秒；复用已建立的连接避免每个路径点重复等待。
            with self.ik_lock:
                try:
                    if self.ik_service is None:
                        self.ik_service = rospy.ServiceProxy(self.ik_name, SolvePositionIK, persistent=True)
                    answers.put((self.ik_service(request), None))
                except Exception as exc:
                    if self.ik_service is not None:
                        self.ik_service.close()
                        self.ik_service = None
                    answers.put((None, exc))
        threading.Thread(target=request_service, daemon=True).start()
        deadline = time.monotonic()+timeout
        while True:
            self.check_cancel()
            if time.monotonic() >= deadline:
                raise RuntimeError('Baxter IK 服务响应超时')
            try:
                response, error = answers.get(timeout=.05)
                break
            except queue.Empty:
                continue
        if error:
            raise RuntimeError(f'Baxter IK 服务调用失败：{error}')
        return ik_solution(response)

    def solve_path(self, start, goal, quat, seed):
        points = cartesian_points(start, goal, self.step)
        print(f'正在预检全部 {len(points)} 个路径点的 IK（首次连接可能需要约 10 秒）', flush=True)
        path = []
        for index, point in enumerate(points):
            self.check_cancel()
            angles = self.call_ik(point, quat, seed)
            jump = max(abs(angles[name]-seed[name]) for name in seed)
            if jump > .5:
                raise ValueError(f'第 {index+1} 段 IK 关节跳变 {jump:.2f} rad，请调整手臂姿态后重新选点')
            path.append((point, angles))
            seed = angles
        return path

    def stop(self, announce=True):
        self.cancel.set()
        with self.lock:
            if self.execute and self.limb is not None and self.moving.is_set():
                try:
                    self.limb.set_joint_positions(self.current_joints())
                except Exception as exc:
                    print(f'保持当前位置指令未发出：{exc}', flush=True)
            if self.execute and self.grip_pub is not None and self.grip is not None:
                self.publish_gripper(EndEffectorCommand.CMD_STOP, {})
        if announce:
            print('已取消后续命令并请求保持当前位置；需要立即切断运动时使用机器人急停。', flush=True)

    def move_to(self, goal, quat, expected, record_plan=None):
        self.ensure_limb()
        start, orientation = self.current()
        wanted = Rotation.from_quat(quat)
        if np.linalg.norm(start-expected) > .01 or (orientation.inv()*wanted).magnitude() > np.deg2rad(2):
            raise ValueError('右臂已离开选点/上一步姿态，请重新选点')
        initial_joints = self.current_joints()
        path = self.solve_path(start, np.asarray(goal), quat, initial_joints)
        self.require_enabled()
        now, rotation = self.current()
        if (np.linalg.norm(now-start) > .005 or (rotation.inv()*orientation).magnitude() > np.deg2rad(1)
                or max(abs(v-initial_joints[k]) for k, v in self.current_joints().items()) > .02):
            raise ValueError('IK 计算期间手臂发生移动，请重新选点')
        if record_plan:
            record_plan([{'endpoint_m': point.tolist(), 'joints_rad': angles} for point, angles in path])
        self.moving.set()
        try:
            with self.lock:
                self.check_cancel()
                self.limb.set_joint_position_speed(self.speed)
            for index, (point, angles) in enumerate(path):
                self.require_enabled()
                print(f'执行路径 {index+1}/{len(path)}，末端目标(m)：{point.round(4).tolist()}', flush=True)
                self.move_segment(angles)
            self.wait_reached(goal, wanted)
        except Exception:
            self.stop(announce=False)
            raise
        finally:
            self.moving.clear()
            self.limb.set_joint_position_speed(.3)

    def move_segment(self, angles):
        # SDK 的正常关节位置模式；单调时钟超时，并在与停止指令共用的锁内发命令。
        # 这样 x 后不会由另一个线程继续重发旧目标，也不依赖旧机器人时间戳。
        filtered = self.current_joints()
        deadline = time.monotonic()+self.timeout
        alpha = 1-np.exp(-2*np.pi*.2*.01)
        while True:
            self.require_enabled()
            actual = self.current_joints()
            if max(abs(actual[name]-value) for name, value in angles.items()) <= .004:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('关节运动超时且实际未到位，取消后续路径')
            for name in angles:
                filtered[name] += alpha*(angles[name]-filtered[name])
            with self.lock:
                self.check_cancel()
                self.limb.set_joint_positions(dict(filtered))
            self.cancel.wait(.01)

    def wait_reached(self, goal, wanted):
        deadline = time.monotonic()+3
        consecutive = 0
        while time.monotonic() < deadline:
            self.require_enabled()
            actual, rotation = self.current()
            good = np.linalg.norm(actual-goal) <= .005 and (rotation.inv()*wanted).magnitude() <= np.deg2rad(1.5)
            consecutive = consecutive+1 if good else 0
            if consecutive >= 3:
                return
            self.cancel.wait(.05)
        raise RuntimeError('末端位置尚未到达 5 mm / 1.5° 内，未报告成功')

    def go_to_target(self, selection, clearance, inset, save_plan, stop_above_m=.007):
        reported, rotation = self.current()
        plan = target_plan(selection, reported, rotation, self.offset, clearance, inset, stop_above_m)
        plan['orientation_sampled_at'] = 'move key p'
        print('按 p 时当前姿态 [qx,qy,qz,qw]：', plan['quat'], flush=True)
        print(f'目标沿基座 Z 轴抬高 {stop_above_m*1000:g} mm（减少下降量）', flush=True)
        print('夹爪点目标(m)：', plan['grasp'], flush=True)
        print('发送给 Baxter endpoint 的目标(m)：', plan['grasp_reported'], flush=True)
        def record(waypoints):
            plan['waypoints'] = waypoints
            save_plan(plan)
        self.move_to(np.array(plan['grasp_reported']), plan['quat'], reported, record)
        return plan

    def publish_gripper(self, command, arguments):
        self.sequence += 1
        msg = EndEffectorCommand(id=self.grip.id, command=command, args=json.dumps(arguments),
                                 sender=rospy.get_name(), sequence=self.sequence)
        self.grip_pub.publish(msg)
        return msg

    def gripper(self, opening):
        if not self.execute:
            raise RuntimeError('预览模式不允许夹爪动作')
        self.current()
        self.wait(lambda: self.robot is not None and self.grip is not None and self.grip_properties is not None,
                  45, '收不到右臂夹爪状态或属性')
        self.require_enabled()
        self.require_fresh(('grip',))
        if self.grip_properties.ui_type != EndEffectorProperties.ELECTRIC_GRIPPER:
            raise ValueError('该开合操作仅适用于 Baxter 电动夹爪')
        if self.grip.calibrated != EndEffectorState.STATE_TRUE or self.grip.error != EndEffectorState.STATE_FALSE:
            raise ValueError('右侧夹爪未完成自身标定或有故障，请先检查夹爪')
        if self.grip.ready != EndEffectorState.STATE_TRUE:
            raise ValueError('右侧夹爪尚未准备好接收新动作')
        self.wait(lambda: self.grip_pub.get_num_connections() > 0, 5, '右侧夹爪命令接口未连接')
        with self.lock:
            self.check_cancel()
            issued = time.monotonic()
            msg = self.publish_gripper(EndEffectorCommand.CMD_GO, {'position': 100.0 if opening else 0.0})
        deadline = time.monotonic()+8
        acknowledged = False
        while time.monotonic() < deadline:
            self.require_enabled()
            self.require_fresh(('grip',))
            state = self.grip
            acknowledged = acknowledged or (self.times.get('grip', 0) >= issued
                                             and state.command == msg.command
                                             and state.command_sender == msg.sender
                                             and state.command_sequence in (msg.sequence, 0))
            done = abs(state.position-(100 if opening else 0)) < 5 or (not opening and state.gripping == 1)
            if acknowledged and state.moving == 0 and done:
                print(f'夹爪{"张开" if opening else "闭合"}完成；反馈 gripping={state.gripping}，position={state.position:.1f}', flush=True)
                return
            if state.error != 0:
                raise RuntimeError('夹爪报告故障')
            self.cancel.wait(.05)
        raise RuntimeError('夹爪动作反馈超时，请观察实际夹爪状态')


def ik_solution(response):
    """兼容 rospy 对 uint8[] 的 bytes/list 返回形式和旧版本 isValid 字段。"""
    result = response.result_type
    if len(result):
        value = result[0]
        if isinstance(value, str):
            value = ord(value)
        valid = int(value) != 0
    else:
        valid = bool(len(response.isValid) and response.isValid[0])
    if not valid or len(response.joints) != 1:
        raise ValueError('目标姿态 IK 无解；请调整当前右臂朝向或选取可达点')
    state = response.joints[0]
    if len(state.name) != 7 or len(state.position) != 7 or len(set(state.name)) != 7:
        raise ValueError('IK 响应未包含七个不同的右臂关节')
    return validate_action({'format': 'baxter_joint_sample_v1', 'arm': 'right',
                            'joint_positions_rad': dict(zip(state.name, state.position))}, 'right')


def load_calibration(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    intr = json.loads(path.with_name('intrinsics.json').read_text(encoding='utf-8'))
    validate_intrinsics(intr)
    transform = validate_transform(result['T_base_camera'])
    if not np.allclose(result['camera_matrix'], intr['camera_matrix']) or not np.allclose(result['dist_coeffs'], intr['dist_coeffs']):
        raise ValueError('标定结果与同目录相机内参不匹配')
    session = json.loads(path.with_name('session.json').read_text(encoding='utf-8'))
    if session.get('robot') != 'Baxter' or session.get('arm') != 'right':
        raise ValueError('请选择由 Baxter 右臂采集的数据所计算的标定文件')
    return transform, intr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=Path(__file__).parent/'1/hand_eye_result.json')
    parser.add_argument('--namespace', default='/baxter_camera')
    parser.add_argument('--execute', action='store_true', help='允许 p/o/c/r 按键执行动作')
    parser.add_argument('--headless-check', action='store_true', help='只读检查图像/深度/右臂和当前姿态 IK 后退出')
    parser.add_argument('--speed', type=float, default=.1, help='SDK 关节速度比例，默认 0.1')
    parser.add_argument('--step', type=float, default=.04, help='笛卡尔路径 IK 采样间距，米，默认 .04')
    parser.add_argument('--move-timeout', type=float, default=20, help='每段关节运动到位超时，秒')
    parser.add_argument('--clearance', type=float, default=.08, help='按 r 退回距离，米')
    parser.add_argument('--grasp-inset', type=float, default=0, help='沿起点到目标方向深入表面距离，米')
    parser.add_argument('--stop-above-mm', type=float, default=7,
                        help='目标沿基座 Z 轴抬高的距离，毫米；默认 7，即少下降 7 mm')
    parser.add_argument('--tcp-offset-mm', type=float, nargs=3, default=[0, 0, 0], metavar=('X', 'Y', 'Z'),
                        help='物理夹爪点相对 Baxter endpoint 的额外偏移；默认 0 0 0')
    args = parser.parse_args(rospy.myargv()[1:])
    if (not .01 <= args.speed <= .3 or not .01 <= args.step <= .1 or not .02 <= args.clearance <= .2
            or not 0 <= args.grasp_inset <= .05 or not 1 <= args.move_timeout <= 120):
        parser.error('speed=.01–.3，step=.01–.1 m，clearance=.02–.2 m，grasp-inset=0–.05 m，move-timeout=1–120 s')
    if not np.isfinite(args.tcp_offset_mm).all() or np.linalg.norm(args.tcp_offset_mm) > 300:
        parser.error('tcp-offset-mm 必须是有限数值且总长度不超过 300 mm')
    if not np.isfinite(args.stop_above_mm) or not 0 <= args.stop_above_mm <= 100:
        parser.error('stop-above-mm 必须是 0–100 的有限数值')
    if args.headless_check and args.execute:
        parser.error('--headless-check 为只读检查，不能与 --execute 同时使用')
    return args


def main():
    args = parse_args()
    transform, intrinsics = load_calibration(args.calibration)
    if not args.headless_check and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise SystemExit('图像交互需在本地图形桌面终端运行；可用 --headless-check 检查接口')
    rospy.init_node('baxter_click_grasp', anonymous=True)
    camera = Camera('/'+args.namespace.strip('/'), intrinsics)
    arm = RightArm(args.execute, args.speed, np.array(args.tcp_offset_mm)/1000, args.step, args.move_timeout)
    if args.headless_check:
        arm.current()
        deadline = time.monotonic()+45
        while camera.snapshot() is None and time.monotonic() < deadline:
            arm.check_cancel()
            arm.cancel.wait(.05)
        bundle = camera.snapshot()
        if bundle is None or time.monotonic()-bundle[2] > 1:
            raise RuntimeError('收不到同步彩色图、对齐深度和匹配的 CameraInfo')
        xyz, orientation = arm.current()
        solution = arm.call_ik(xyz, orientation.as_quat(), arm.current_joints())
        repeat_started = time.monotonic()
        arm.call_ik(xyz, orientation.as_quat(), arm.current_joints())
        print(f'IK 连接复用检查通过，第二次查询耗时 {time.monotonic()-repeat_started:.3f} s', flush=True)
        depth = bundle[1]
        print(f'只读检查通过：RGB={bundle[0].shape}；depth={depth.shape} {depth.dtype}，毫米；有效深度像素={np.count_nonzero(depth)}', flush=True)
        print('当前右臂位置(m)：', xyz.tolist(), flush=True)
        print('当前右臂姿态 IK 有解：', solution, flush=True)
        rospy.signal_shutdown('read-only check completed')
        return
    log_dir = Path(__file__).parent / datetime.now().strftime('grasp_baxter_%Y%m%d_%H%M%S_%f')
    log_dir.mkdir()
    (log_dir/'run.json').write_text(json.dumps(dict(calibration=str(args.calibration.resolve()),
        execute=args.execute, namespace=args.namespace, tcp_offset_in_endpoint_mm=args.tcp_offset_mm,
        orientation_source='fresh right endpoint pose at p', speed=args.speed, step_m=args.step,
        clearance_m=args.clearance, grasp_inset_m=args.grasp_inset,
        stop_above_base_z_mm=args.stop_above_mm), indent=2), encoding='utf-8')
    print(f'模式：{"按键执行" if args.execute else "仅预览"}；日志：{log_dir}', flush=True)
    print('左键选点；p 保持当前姿态到目标；o 张开；c 闭合；r 退回；x 停止；q 退出。', flush=True)
    print('Baxter endpoint 已包含所配置的夹爪长度；额外偏移(mm)：', args.tcp_offset_mm, flush=True)
    print(f'目标沿基座 Z 轴抬高 {args.stop_above_mm:g} mm，减少相应下降量。', flush=True)
    print('保持相机和机器人底座固定；先手动调好右臂朝向。路径以小段 IK+SDK 关节运动执行，没有环境碰撞规划。', flush=True)
    run_viewer(args, camera, arm, transform, intrinsics, log_dir)


def run_viewer(args, camera, arm, transform, intrinsics, log_dir):
    jobs = queue.Queue()
    busy = False
    generation = 0
    target = None
    stage = 'select a point'
    display_bundle = None
    window = 'Baxter Right Grasp | RGB / aligned depth'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def save_json(name, value):
        filename = datetime.now().strftime('%H%M%S_%f')+'_'+name+'.json'
        (log_dir/filename).write_text(json.dumps(value, indent=2), encoding='utf-8')

    def start_task(action, task):
        nonlocal busy
        if busy:
            return
        busy = True
        arm.cancel.clear()
        token = generation
        def run():
            try:
                jobs.put((token, action, task(), None))
            except Exception as exc:
                if args.execute and action != 'select':
                    arm.stop(announce=False)
                jobs.put((token, action, None, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def select(event, u, v, flags, param):
        nonlocal target, stage
        if event != cv2.EVENT_LBUTTONDOWN or busy or display_bundle is None:
            return
        if u >= display_bundle[0].shape[1]:
            print('请点击左侧 RGB 图像', flush=True)
            return
        if stage in ('at grasp', 'closed'):
            print('先按 r 退回，再重新选点', flush=True)
            return
        bundle = display_bundle
        if time.monotonic()-bundle[2] > 1:
            print('图像已过期，请等待新图像', flush=True)
            return
        target, stage = None, 'selecting'
        def prepare():
            camera_xyz, surface = point_from_depth(u, v, bundle[1], bundle[5], transform)
            reported, rotation = arm.current()
            if time.monotonic()-bundle[2] > 1:
                raise ValueError('读取手臂状态时选点图像已过期，请重新点击')
            selection = {'pixel': [u, v], 'camera_xyz': camera_xyz.tolist(), 'surface': surface.tolist(),
                         'camera_matrix_used': bundle[5]['camera_matrix'],
                         'dist_coeffs_used': bundle[5]['dist_coeffs'],
                         'stamp': bundle[3], 'frame_id': bundle[4], 'selected_monotonic': time.monotonic()}
            selection = target_plan(selection, reported, rotation, arm.offset, args.clearance, args.grasp_inset,
                                    args.stop_above_mm/1000)
            name = datetime.now().strftime('%H%M%S_%f')
            if (not cv2.imwrite(str(log_dir/f'{name}_color.png'), bundle[0])
                    or not cv2.imwrite(str(log_dir/f'{name}_depth.png'), bundle[1])):
                raise RuntimeError('选点图像保存失败')
            save_json('target', selection)
            print('像素：', selection['pixel'], '相机坐标(m)：', selection['camera_xyz'], flush=True)
            print('选点表面在 Baxter 基座下坐标(m)：', selection['surface'], flush=True)
            print(f'抬高 {args.stop_above_mm:g} mm 后夹爪点目标(m)：', selection['grasp'], flush=True)
            return selection
        start_task('select', prepare)
    cv2.setMouseCallback(window, select)
    try:
        while not rospy.is_shutdown():
            try:
                token, action, value, error = jobs.get_nowait()
                busy = False
                if token == generation:
                    with (log_dir/'actions.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'time': datetime.now().isoformat(), 'action': action, 'error': error})+'\n')
                    if error:
                        print('操作未完成：'+error, flush=True)
                        target, stage = None, 'error - reselect'
                    elif action == 'select':
                        target, stage = value, 'selected'
                    elif action == 'move_target':
                        target, stage = value, 'at grasp'
                        print('已到达目标，可按 c 闭合夹爪或 r 退回', flush=True)
                    elif action == 'open':
                        target, stage = None, 'opened - select point'
                    else:
                        stage = {'close': 'closed', 'retreat': 'retreated'}[action]
                        print('完成：'+stage, flush=True)
            except queue.Empty:
                pass
            bundle = camera.snapshot()
            display_bundle = bundle
            if bundle is None:
                image = np.zeros((intrinsics['height'], intrinsics['width']*2, 3), np.uint8)
                age = float('inf')
            else:
                color = bundle[0].copy()
                if target:
                    cv2.circle(color, tuple(target['pixel']), 5, (0, 0, 255), 2)
                depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(bundle[1], alpha=.04), cv2.COLORMAP_JET)
                depth_vis[bundle[1] == 0] = 0
                image = np.hstack([color, depth_vis])
                age = time.monotonic()-bundle[2]
            mode = 'EXECUTE' if args.execute else 'PREVIEW'
            cv2.putText(image, f'{mode} | {stage} | {"BUSY" if busy else "READY"}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.putText(image, f'frame age={age:.1f}s | o open / p move / c close / r retreat / x stop / q quit', (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 255, 0), 1)
            if target:
                cv2.putText(image, f'target XYZ(m): {np.round(target["grasp"], 3)} | Z+{args.stop_above_mm:g}mm', (8, 68), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('x'):
                generation += 1
                arm.stop()
                target, stage = None, 'stopped - reselect'
                continue
            if busy or key not in map(ord, 'opcr'):
                continue
            if not args.execute:
                print('当前是预览模式；启动时加 --execute 才发送动作', flush=True)
                continue
            if key == ord('o'):
                start_task('open', lambda: arm.gripper(True))
                continue
            if target is None:
                print('先在左侧 RGB 图像点击目标点', flush=True)
                continue
            if key == ord('p') and stage == 'selected':
                if age > 1 or time.monotonic()-target['selected_monotonic'] > 60:
                    print('图像/目标已过期，请重新选点', flush=True)
                    continue
                selection = dict(target)
                start_task('move_target', lambda: arm.go_to_target(selection, args.clearance, args.grasp_inset,
                                                                  lambda plan: save_json('move_plan', plan),
                                                                  args.stop_above_mm/1000))
            elif key == ord('c') and stage == 'at grasp':
                if age > 1:
                    print('图像已过期，请恢复画面后再闭合', flush=True)
                    continue
                plan = dict(target)
                def close_at_target():
                    xyz, rotation = arm.current()
                    if (np.linalg.norm(xyz-np.array(plan['grasp_reported'])) > .005
                            or (rotation.inv()*Rotation.from_quat(plan['quat'])).magnitude() > np.deg2rad(2)):
                        raise ValueError('右臂已离开抓取位置，取消夹爪闭合')
                    arm.gripper(False)
                start_task('close', close_at_target)
            elif key == ord('r') and stage in ('at grasp', 'closed'):
                plan = dict(target)
                start_task('retreat', lambda: arm.move_to(np.array(plan['retreat_reported']), plan['quat'],
                                                         np.array(plan['grasp_reported']),
                                                         lambda waypoints: save_json('retreat_plan', {'waypoints': waypoints})))
            else:
                print('顺序：o 张开 → 选点 → p 到目标 → c 闭合（可选）→ r 退回', flush=True)
    finally:
        if busy:
            arm.stop(announce=False)
        else:
            arm.cancel.set()
        rospy.signal_shutdown('viewer closed')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))
