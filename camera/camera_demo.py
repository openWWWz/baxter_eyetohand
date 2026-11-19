#!/usr/bin/env python3
"""独立相机预览：自动选择 RealSense/USB，或复用已有 ROS 画面；不加载标定。"""
import argparse
import os
import threading
import time
from pathlib import Path

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
if Path(FONT_DIR).is_dir():
    os.environ.setdefault('QT_QPA_FONTDIR', FONT_DIR)
import cv2
import numpy as np
if Path(FONT_DIR).is_dir():
    os.environ['QT_QPA_FONTDIR'] = FONT_DIR


WINDOW = 'Camera Demo | RGB / Depth | q: quit'


def usb_devices():
    """按编号扫描所有视频节点，名字仅用于展示和避免把深度节点当成 USB 彩色图。"""
    result = []
    for path in sorted(Path('/dev').glob('video[0-9]*'), key=lambda p: int(p.name[5:])):
        name_file = Path('/sys/class/video4linux') / path.name / 'name'
        name = name_file.read_text().strip() if name_file.exists() else path.name
        result.append((str(path), name))
    return result


def publisher_running():
    try:
        pid = int(Path(__file__).with_name('publisher.pid').read_text().strip())
        return b'publish_realsense_ros.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (OSError, ValueError):
        return False


def decode_image(msg, depth=False):
    """直接解码 ROS Image，无 cv_bridge/内参依赖；支持行填充与深度字节序。"""
    if depth:
        if msg.encoding not in ('16UC1', '32FC1'):
            raise ValueError(f'不支持的深度编码：{msg.encoding}')
        dtype = np.dtype(('>' if msg.is_bigendian else '<') + ('u2' if msg.encoding == '16UC1' else 'f4'))
        if msg.step < msg.width*dtype.itemsize or len(msg.data) != msg.height*msg.step:
            raise ValueError('深度图长度或步幅无效')
        array = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                           strides=(msg.step, dtype.itemsize)).astype(np.float32)
        return array * (.001 if msg.encoding == '16UC1' else 1.)
    formats = {'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR),
               'mono8': (1, cv2.COLOR_GRAY2BGR), 'bgra8': (4, cv2.COLOR_BGRA2BGR),
               'rgba8': (4, cv2.COLOR_RGBA2BGR)}
    if msg.encoding not in formats:
        raise ValueError(f'不支持的彩色编码：{msg.encoding}')
    channels, conversion = formats[msg.encoding]
    if msg.step < msg.width*channels or len(msg.data) != msg.height*msg.step:
        raise ValueError('彩色图长度或步幅无效')
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    array = rows[:, :msg.width*channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        array = array[:, :, 0]
    return cv2.cvtColor(array, conversion) if conversion is not None else array.copy()


class RosCamera:
    def __init__(self, args):
        try:
            import rospy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError('读取已有 ROS 画面请用 bash run_camera_demo.sh 自动加载环境') from exc
        self.rospy = rospy
        self.condition = threading.Condition()
        self.color = self.depth = None
        self.color_time = self.depth_time = 0
        self.seq = self.last = 0
        self.label = 'ROS '+args.rgb_topic
        if not rospy.core.is_initialized():
            rospy.init_node('standalone_camera_demo', anonymous=True, disable_signals=True)
        self.subs = [rospy.Subscriber(args.rgb_topic, Image, self.on_color, queue_size=1, buff_size=2**23),
                     rospy.Subscriber(args.depth_topic, Image, self.on_depth, queue_size=1, buff_size=2**23)]
        try:
            self.first = self.read(timeout=10)
        except Exception:
            self.close()
            raise

    def on_color(self, msg):
        try:
            array = decode_image(msg)
            with self.condition:
                self.color, self.color_time = array, time.monotonic()
                self.seq += 1
                self.condition.notify_all()
        except Exception as exc:
            self.rospy.logwarn_throttle(5, '彩色解码失败：%s', exc)

    def on_depth(self, msg):
        try:
            array = decode_image(msg, depth=True)
            with self.condition:
                self.depth, self.depth_time = array, time.monotonic()
        except Exception as exc:
            self.rospy.logwarn_throttle(5, '深度解码失败：%s', exc)

    def read(self, timeout=2):
        if getattr(self, 'first', None) is not None:
            first, self.first = self.first, None
            return first
        deadline = time.monotonic()+timeout
        with self.condition:
            while self.seq == self.last:
                if self.rospy.is_shutdown() or time.monotonic() >= deadline:
                    raise RuntimeError('收不到新的 ROS 彩色画面；请检查相机发布器或改用 --source realsense/usb')
                self.condition.wait(.05)
            self.last = self.seq
            depth = self.depth if abs(self.color_time-self.depth_time) < .5 else None
            return self.color, depth

    def close(self):
        for sub in self.subs:
            sub.unregister()


class RealSenseCamera:
    def __init__(self, args):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('当前 Python 环境未安装 pyrealsense2；请使用 eyetohand 环境') from exc
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError('没有检测到 RealSense')
        # 每次枚举实际接入的设备，没有固定序列号。
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        self.label = device.get_info(rs.camera_info.name)+' / '+serial
        self.pipeline = rs.pipeline()
        self.started = False
        self.align = rs.align(rs.stream.color)
        errors = []
        # 先尝试常用彩色+深度；再让 SDK 自动选择分辨率；普通彩色也可单独工作。
        for exact, depth in ((True, True), (False, True), (False, False)):
            cfg = rs.config()
            cfg.enable_device(serial)
            if exact:
                cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
                cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
            else:
                cfg.enable_stream(rs.stream.color, rs.format.bgr8)
                if depth:
                    cfg.enable_stream(rs.stream.depth, rs.format.z16)
            try:
                profile = self.pipeline.start(cfg)
                self.started = True
                self.has_depth = depth
                self.scale = profile.get_device().first_depth_sensor().get_depth_scale() if depth else 0
                return
            except RuntimeError as exc:
                errors.append(str(exc))
                if self.started:
                    self.close()
                    raise
        raise RuntimeError('RealSense 打开失败（可能被其他程序占用）：'+errors[-1])

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        if self.has_depth:
            frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color:
            raise RuntimeError('RealSense 没有返回彩色帧')
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32)*self.scale if depth else None
        return rgb, depth_m

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


class UsbCamera:
    def __init__(self, args):
        candidates = [(args.device, '手动选择的 USB 相机')] if args.device else [
            (path, name) for path, name in usb_devices() if 'realsense' not in name.lower()]
        for path, name in candidates:
            device = int(path) if str(path).isdigit() else path
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap, self.first, self.label = cap, frame, f'{name} / {path}'
                return
            cap.release()
        raise RuntimeError('找不到可用 USB 彩色相机；可用 --list 查看，--source usb --device /dev/videoN 指定')

    def read(self):
        if self.first is not None:
            frame, self.first = self.first, None
            return frame, None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError('USB 相机没有返回新画面，请检查连接或设备占用')
        return frame, None

    def close(self):
        self.cap.release()


def open_camera(args):
    if args.device:
        modes = ['usb']
    elif args.source == 'auto':
        modes = ['ros', 'realsense', 'usb'] if publisher_running() else ['realsense', 'usb']
    else:
        modes = [args.source]
    types = {'ros': RosCamera, 'realsense': RealSenseCamera, 'usb': UsbCamera}
    errors = []
    for mode in modes:
        print('正在读取：'+mode, flush=True)
        try:
            return types[mode](args)
        except (RuntimeError, OSError) as exc:
            errors.append(f'{mode}: {exc}')
            print(errors[-1], flush=True)
    raise RuntimeError('\n'.join(errors))


def preview_image(color, depth):
    if depth is None:
        return color.copy()
    valid = np.isfinite(depth) & (depth > 0)
    visible = np.where(valid, depth, 0)
    depth_view = cv2.applyColorMap(np.clip(visible * 255/2, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_view[~valid] = 0
    if depth_view.shape[:2] != color.shape[:2]:
        depth_view = cv2.resize(depth_view, (color.shape[1], color.shape[0]))
    return np.hstack([color, depth_view])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['auto', 'realsense', 'usb', 'ros'], default='auto')
    parser.add_argument('--device', help='指定 USB 设备，例如 /dev/video6 或 6；不传则自动选')
    parser.add_argument('--list', action='store_true', help='列出本机相机设备后退出')
    parser.add_argument('--check', action='store_true', help='读取 3 帧并输出尺寸，不打开窗口')
    parser.add_argument('--frames', type=int, default=0, help='读取指定帧数后退出，0 为持续预览')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--rgb-topic', default='/baxter_camera/color/image_raw')
    parser.add_argument('--depth-topic', default='/baxter_camera/aligned_depth_to_color/image_raw')
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.frames < 0:
        parser.error('分辨率和 fps 必须为正数，frames 必须非负')
    if args.list:
        try:
            import pyrealsense2 as rs
            for device in rs.context().query_devices():
                print('RealSense:', device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.serial_number))
        except ImportError:
            print('未安装 pyrealsense2；USB 设备仍可读取')
        for path, name in usb_devices():
            print('USB:', path, name)
        return
    if not args.check and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('请在本地图形桌面终端运行；--check 可只检查图像')
    camera = open_camera(args)
    print('当前相机：'+camera.label, flush=True)
    print('左侧彩色 / 右侧深度（如有）；q 或 Esc 退出。不读取标定、不控制机器人。', flush=True)
    count = 0
    maximum = 3 if args.check else args.frames
    try:
        if not args.check:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        started = time.monotonic()
        while True:
            rgb, depth = camera.read()
            count += 1
            if args.check:
                print(f'帧 {count}: RGB={rgb.shape}, depth={None if depth is None else depth.shape}', flush=True)
            else:
                image = preview_image(rgb, depth)
                rate = count/max(time.monotonic()-started, .001)
                cv2.putText(image, f'RGB {rgb.shape[1]}x{rgb.shape[0]} | {rate:.1f} FPS | q: quit',
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if maximum and count >= maximum:
                break
    finally:
        camera.close()
        if not args.check:
            cv2.destroyAllWindows()
    print(f'已读取 {count} 帧，正常退出。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as exc:
        raise SystemExit(str(exc))
