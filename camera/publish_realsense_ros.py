#!/usr/bin/env python3
"""把本机 RealSense 彩色帧及对齐深度发布为 ROS1 话题。"""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


def camera_info(intr, header):
    """发布矫正后彩色图的针孔内参；对齐深度使用同一套内参。"""
    msg = CameraInfo()
    msg.header = header
    msg.width, msg.height = intr.width, intr.height
    msg.distortion_model = 'plumb_bob'
    msg.D = [0.0] * 5
    msg.K = [intr.fx, 0, intr.ppx, 0, intr.fy, intr.ppy, 0, 0, 1]
    msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    msg.P = [intr.fx, 0, intr.ppx, 0, 0, intr.fy, intr.ppy, 0, 0, 0, 1, 0]
    return msg
                                                                                                                                        

def image_msg(array, encoding, header):
    array = np.ascontiguousarray(array)
    msg = Image()
    msg.header = header
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


def rectification_maps(intr):
    """RealSense 原始像素经过校正后，输出图像的 D=0。"""
    model = str(intr.model)
    if 'inverse_brown_conrady' not in model and 'brown_conrady' not in model and 'none' not in model:
        raise ValueError(f'彩色畸变模型暂不支持：{model}')
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist, np.eye(3), K, (intr.width, intr.height), cv2.CV_32FC1
    )
    return map_x, map_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', help='可选：指定 RealSense 设备；默认由 SDK 自动选择')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--namespace', default='/baxter_camera', help='话题前缀，默认 /baxter_camera')
    parser.add_argument('--preview', action='store_true', help='显示彩色/对齐深度预览，按 q 退出')
    args = parser.parse_args(rospy.myargv()[1:])
    if min(args.width, args.height, args.fps) <= 0:
        parser.error('width/height/fps 必须为正数')

    namespace = '/' + args.namespace.strip('/')
    rospy.init_node('baxter_fixed_realsense_camera', anonymous=True)
    color_pub = rospy.Publisher(namespace + '/color/image_raw', Image, queue_size=1)
    color_info_pub = rospy.Publisher(namespace + '/color/camera_info', CameraInfo, queue_size=1)
    depth_pub = rospy.Publisher(namespace + '/aligned_depth_to_color/image_raw', Image, queue_size=1)
    depth_info_pub = rospy.Publisher(namespace + '/aligned_depth_to_color/camera_info', CameraInfo, queue_size=1)

    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    started = False
    try:
        profile = pipeline.start(cfg)
        started = True
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        map_x, map_y = rectification_maps(intr)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        frame_id = 'baxter_fixed_camera_color_optical_frame'
        rospy.loginfo('RealSense；彩色 %sx%s@%s，深度对齐到彩色；depth 16UC1 mm',
                      intr.width, intr.height, args.fps)
        rospy.loginfo('RGB: %s/color/image_raw；内参: %s/color/camera_info', namespace, namespace)
        rospy.loginfo('对齐深度: %s/aligned_depth_to_color/image_raw', namespace)
        seq = 0
        last_log = time.monotonic()
        last_frame = time.monotonic()
        while not rospy.is_shutdown():
            try:
                frames = align.process(pipeline.wait_for_frames(timeout_ms=2500))
                color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    rospy.logwarn_throttle(5, '等待同一组彩色帧和对齐深度帧')
                    continue
                raw_color = np.asanyarray(color_frame.get_data())
                raw_depth = np.asanyarray(depth_frame.get_data())
                rgb = cv2.remap(raw_color, map_x, map_y, cv2.INTER_LINEAR)
                aligned_depth = cv2.remap(raw_depth, map_x, map_y, cv2.INTER_NEAREST)
                if not np.isclose(depth_scale, 0.001):
                    aligned_depth = np.clip(np.rint(aligned_depth.astype(np.float32) *
                                                     depth_scale * 1000), 0, 65535).astype(np.uint16)
                header = Header(seq=seq, stamp=rospy.Time.now(), frame_id=frame_id)
                color_pub.publish(image_msg(rgb, 'bgr8', header))
                color_info_pub.publish(camera_info(intr, header))
                depth_pub.publish(image_msg(aligned_depth, '16UC1', header))
                depth_info_pub.publish(camera_info(intr, header))
                last_frame = time.monotonic()
                seq += 1
                if args.preview:
                    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth, alpha=0.04), cv2.COLORMAP_JET)
                    view = np.hstack((rgb, depth_vis))
                    cv2.imshow('Baxter fixed camera: RGB | aligned depth (q to exit)', view)
                    if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                        break
                if time.monotonic() - last_log > 10:
                    rospy.loginfo('相机发布正常：%d 帧', seq)
                    last_log = time.monotonic()
            except RuntimeError as exc:
                if rospy.is_shutdown():
                    break
                rospy.logwarn_throttle(5, 'RealSense 帧读取超时：%s', exc)
                if time.monotonic() - last_frame > 15:
                    raise RuntimeError('RealSense 连续 15 秒没有有效帧；可能已断开或更换。'
                                       '请重新运行 python publish_realsense_ros.py --fps 15。') from exc
    finally:
        if started:
            try:
                pipeline.stop()
            except RuntimeError:
                pass  # 设备断开时 SDK 可能已自动停止 pipeline。
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
