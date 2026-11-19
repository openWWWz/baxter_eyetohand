"""离线抓取验证：深度单位、TCP、无解拒绝、未到位和取消；不连接机器人。"""
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Header
from baxter_core_msgs.msg import AssemblyState, EndEffectorCommand, EndEffectorState, EndEffectorProperties
import rospy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baxter_grasp_geometry import (cartesian_points, depth_image, point_from_depth,
                                    target_plan, validate_camera_info, validate_transform)
from test_grasp_baxter import Camera, RightArm, ik_solution, load_calibration, parse_args

INTR = {'width': 20, 'height': 20,
        'camera_matrix': [[100., 0, 10], [0, 100., 10], [0, 0, 1]], 'dist_coeffs': [0.] * 5}
JOINTS = dict(zip(('right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2'),
                  [.1, -.7, .1, 1.2, .1, .3, .1]))


def camera_info():
    return CameraInfo(width=20, height=20, K=np.asarray(INTR['camera_matrix']).ravel().tolist(),
                      D=[0.]*5, distortion_model='plumb_bob',
                      header=Header(stamp=rospy.Time(1), frame_id='camera_optical'))


def arm():
    a = object.__new__(RightArm)
    a.execute, a.speed, a.offset, a.step, a.timeout = True, .1, np.zeros(3), .04, .01
    a.cancel, a.moving = threading.Event(), threading.Event()
    a.condition, a.lock = threading.Condition(), threading.RLock()
    a.limb, a.grip_pub, a.grip = Mock(), None, None
    a.robot = AssemblyState(enabled=True)
    a.ik_name, a.ik_service, a.ik_lock = '/offline/IKService', None, threading.Lock()
    a.ensure_limb = Mock()
    a.require_enabled = Mock()
    a.current_joints = Mock(return_value=dict(JOINTS))
    a.current = Mock(return_value=(np.array([.4, -.2, .2]), Rotation.identity()))
    return a


class GeometryTests(unittest.TestCase):
    def test_depth_units_and_base_transform(self):
        T = np.eye(4); T[:3, 3] = [.1, .2, .3]
        c, b = point_from_depth(10, 10, np.full((20, 20), 500, np.uint16), INTR, T)
        np.testing.assert_allclose(c, [0, 0, .5])
        np.testing.assert_allclose(b, [.1, .2, .8])
        for depth in (np.zeros((20, 20), np.uint16), np.full((20, 20), 5, np.uint16)):
            with self.assertRaises(ValueError): point_from_depth(10, 10, depth, INTR, T)
        d = np.full((20, 20), 500, np.uint16); d[9, 10] = 900
        with self.assertRaisesRegex(ValueError, '跳变'): point_from_depth(10, 10, d, INTR, T)

    def test_depth_endianness_padding_and_encoding(self):
        msg = Image(height=1, width=2, step=6, encoding='16UC1', is_bigendian=1,
                    data=b'\x01\xf4\x03\xe8\x00\x00')
        np.testing.assert_array_equal(depth_image(msg), [[500, 1000]])
        msg.encoding = '32FC1'
        with self.assertRaises(ValueError): depth_image(msg)

    def test_intrinsics_and_rigid_transform_checks(self):
        info = camera_info(); validate_camera_info(info, INTR)
        info.K[0] += 5
        with self.assertRaises(ValueError): validate_camera_info(info, INTR)
        T = np.eye(4); T[2, 2] = -1
        with self.assertRaises(ValueError): validate_transform(T)

    def test_small_focal_change_uses_current_intrinsics_but_rejects_changed_camera(self):
        info = camera_info()
        info.K[0] = info.K[4] = 99.95
        live = validate_camera_info(info, INTR)
        self.assertEqual(live['camera_matrix'][0][0], 99.95)
        point, _ = point_from_depth(15, 10, np.full((20, 20), 500, np.uint16), live, np.eye(4))
        self.assertAlmostEqual(point[0], 5*.5/99.95)
        self.assertEqual(INTR['camera_matrix'][0][0], 100)
        for change in ('focal', 'principal', 'resolution', 'distortion'):
            with self.subTest(change=change):
                invalid = camera_info()
                if change == 'focal': invalid.K[0] = 100.2
                if change == 'principal': invalid.K[2] += .02
                if change == 'resolution': invalid.width = 21
                if change == 'distortion': invalid.D[0] = .001
                with self.assertRaises(ValueError): validate_camera_info(invalid, INTR)

    def test_current_orientation_and_rotated_tcp_offset(self):
        rotation = Rotation.from_euler('y', 90, degrees=True)
        plan = target_plan({'surface': [.7, 0, 0]}, [.1, 0, 0], rotation,
                           [0, 0, .1], .08, 0)
        np.testing.assert_allclose(plan['start'], [.2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(plan['grasp_reported'], [.6, 0, 0], atol=1e-12)
        np.testing.assert_allclose(plan['retreat_reported'], [.52, 0, 0], atol=1e-12)
        np.testing.assert_allclose(plan['quat'], rotation.as_quat())
        zero = target_plan({'surface': [.7, 0, 0]}, [.1, 0, 0], rotation, [0, 0, 0], .08, 0)
        np.testing.assert_allclose(zero['grasp_reported'], zero['surface'])
        points = cartesian_points([0, 0, 0], [.11, 0, 0], .04)
        self.assertEqual(len(points), 3)
        np.testing.assert_allclose(points[-1], [.11, 0, 0])

    def test_real_calibration_loads(self):
        path = Path(__file__).resolve().parents[1]/'1/hand_eye_result.json'
        if not path.exists(): self.skipTest('本地没有采集数据')
        T, intr = load_calibration(path)
        self.assertEqual(T.shape, (4, 4)); self.assertEqual(intr['width'], 640)

    def test_raise_target_seven_mm_reduces_descent_seventy_to_sixty_three(self):
        selection = {'surface': [.5, -.2, .1]}
        start = [.5, -.2, .17]
        orientation = Rotation.from_euler('xyz', [.3, -.6, .4])
        plan = target_plan(selection, start, orientation, [0, 0, 0], .08, 0, .007)
        self.assertAlmostEqual(start[2]-plan['grasp_reported'][2], .063)
        np.testing.assert_allclose(plan['grasp_reported'], [.5, -.2, .107])
        np.testing.assert_allclose(plan['surface'], selection['surface'])
        np.testing.assert_allclose(plan['retreat'], start)
        # UI 选点预览后按 p 重算，仍只抬高 7 mm。
        again = target_plan(plan, start, orientation, [0, 0, 0], .08, 0, .007)
        np.testing.assert_allclose(again['grasp_reported'], plan['grasp_reported'])

    def test_default_stop_above_and_zero_override(self):
        with patch.object(sys, 'argv', ['test_grasp_baxter.py']):
            self.assertEqual(parse_args().stop_above_mm, 7)
        with patch.object(sys, 'argv', ['test_grasp_baxter.py', '--stop-above-mm', '0']):
            self.assertEqual(parse_args().stop_above_mm, 0)


class CameraTests(unittest.TestCase):
    def test_frame_bundle_carries_live_intrinsics_for_clicked_point(self):
        c = object.__new__(Camera); c.intrinsics = INTR; c.lock = threading.Lock(); c.bundle = None
        info = camera_info(); info.K[0] = info.K[4] = 99.95
        color = Image(height=20, width=20, step=60, encoding='bgr8', data=bytes(1200), header=info.header)
        depth = Image(height=20, width=20, step=40, encoding='16UC1',
                      data=np.full((20, 20), 500, '<u2').tobytes(), header=info.header)
        c.on_pair(color, depth, info)
        self.assertIsNotNone(c.snapshot())
        self.assertEqual(c.snapshot()[5]['camera_matrix'][0][0], 99.95)

    def test_rejects_mixed_frames_timestamps_and_wrong_info(self):
        c = object.__new__(Camera); c.intrinsics = INTR; c.lock = threading.Lock(); c.bundle = None
        info = camera_info()
        color = Image(height=20, width=20, step=60, encoding='bgr8', data=bytes(1200), header=info.header)
        depth = Image(height=20, width=20, step=40, encoding='16UC1',
                      data=np.full((20, 20), 500, '<u2').tobytes(), header=info.header)
        c.on_pair(color, depth, info); self.assertIsNotNone(c.snapshot())
        depth.header = Header(stamp=rospy.Time(1), frame_id='depth_optical')
        with patch('test_grasp_baxter.rospy.logwarn_throttle'): c.on_pair(color, depth, info)
        self.assertIsNone(c.snapshot())


class MotionTests(unittest.TestCase):
    def test_ik_response_bytes_list_and_rejection(self):
        joints = [JointState(name=list(JOINTS), position=list(JOINTS.values()))]
        for kinds in (b'\x01', [2], ''):
            response = SimpleNamespace(result_type=kinds, joints=joints, isValid=[True])
            self.assertEqual(ik_solution(response), JOINTS)
        for resp in [SimpleNamespace(result_type=b'\x00', joints=joints),
                     SimpleNamespace(result_type=b'\x01', joints=[])]:
            with self.assertRaises(ValueError): ik_solution(resp)
        wrong = dict(JOINTS); wrong['right_w1'] = 4
        with self.assertRaises(ValueError):
            ik_solution(SimpleNamespace(result_type=[1], joints=[JointState(name=list(wrong), position=list(wrong.values()))]))

    def test_ik_service_uses_base_frame_user_seed_and_reuses_connection(self):
        a = arm()
        response = SimpleNamespace(result_type=b'\x01', joints=[JointState(name=list(JOINTS), position=list(JOINTS.values()))])
        service = Mock(return_value=response)
        with patch('test_grasp_baxter.rospy.Time.now', return_value=rospy.Time(1)), \
                patch('test_grasp_baxter.rospy.ServiceProxy', return_value=service) as factory:
            a.call_ik([.4, -.2, .2], [0, 0, 0, 1], JOINTS)
            a.call_ik([.4, -.2, .2], [0, 0, 0, 1], JOINTS)
        factory.assert_called_once()
        self.assertTrue(factory.call_args.kwargs['persistent'])
        request = service.call_args.args[0]
        self.assertEqual(request.pose_stamp[0].header.frame_id, 'base')
        self.assertEqual(request.seed_mode, request.SEED_USER)
        self.assertEqual(dict(zip(request.seed_angles[0].name, request.seed_angles[0].position)), JOINTS)

    def test_p_uses_orientation_at_press_instead_of_click(self):
        a = arm(); orientation = Rotation.from_euler('xyz', [.3, .1, -.4])
        a.current.return_value = (np.array([.3, 0, 0]), orientation)
        a.move_to = Mock()
        plan = a.go_to_target({'surface': [.5, 0, 0], 'quat': [0, 0, 0, 1]}, .08, 0, Mock())
        np.testing.assert_allclose(plan['quat'], orientation.as_quat())
        np.testing.assert_allclose(a.move_to.call_args.args[1], orientation.as_quat())
        np.testing.assert_allclose(a.move_to.call_args.args[0], [.5, 0, .007])

    def test_ik_failure_prevents_all_motion(self):
        a = arm(); a.solve_path = Mock(side_effect=ValueError('IK 无解'))
        with self.assertRaises(ValueError): a.move_to(np.array([.5, -.2, .2]), [0, 0, 0, 1], np.array([.4, -.2, .2]))
        a.limb.set_joint_positions.assert_not_called()
        a.limb.set_joint_position_speed.assert_not_called()

    def test_ik_branch_jump_rejected(self):
        a = arm(); a.call_ik = Mock(return_value=dict(JOINTS, right_s0=1.1))
        with self.assertRaisesRegex(ValueError, '跳变'): a.solve_path([.4, 0, 0], [.5, 0, 0], [0, 0, 0, 1], JOINTS)

    def test_timeout_rejects_unreached_and_cancel_sends_no_old_goal(self):
        a = arm(); goal = dict(JOINTS, right_s0=.2)
        with self.assertRaisesRegex(RuntimeError, '超时'): a.move_segment(goal)
        a.cancel.set(); a.limb.reset_mock()
        with self.assertRaisesRegex(RuntimeError, '取消'): a.move_segment(goal)
        a.limb.set_joint_positions.assert_not_called()

    def test_fresh_joint_filter_and_disabled_robot(self):
        a = arm(); del a.require_enabled
        a.times = {k: time.monotonic() for k in ('endpoint', 'joints', 'robot')}
        a.robot = AssemblyState(enabled=False)
        with self.assertRaisesRegex(RuntimeError, '未使能'): a.require_enabled()
        a.robot.enabled = True; a.require_enabled()
        a.times['endpoint'] -= 2
        with self.assertRaisesRegex(RuntimeError, '过期'): a.require_enabled()

    def test_stop_replaces_goal_with_current_joints(self):
        a = arm(); a.moving.set(); a.stop(announce=False)
        self.assertTrue(a.cancel.is_set())
        a.limb.set_joint_positions.assert_called_once_with(JOINTS)

    def test_successful_segment_reads_actual_joint_feedback(self):
        a = arm(); a.timeout = 1
        feedback = dict(JOINTS)
        a.current_joints.side_effect = lambda: dict(feedback)
        a.limb.set_joint_positions.side_effect = lambda values: feedback.update(values)
        a.cancel.wait = Mock()
        goal = dict(JOINTS, right_s0=.15)
        a.move_segment(goal)
        self.assertLessEqual(abs(feedback['right_s0']-.15), .004)
        self.assertGreater(a.limb.set_joint_positions.call_count, 1)

    def test_endpoint_success_requires_position_and_orientation(self):
        a = arm()
        a.current.return_value = (np.array([.4, -.2, .2]), Rotation.identity())
        a.cancel.wait = Mock()
        a.wait_reached(np.array([.4, -.2, .2]), Rotation.identity())
        self.assertGreaterEqual(a.current.call_count, 3)

    def test_open_close_commands_and_hardware_acknowledgement(self):
        a = arm(); a.times = {'grip': time.monotonic()}; a.sequence = 0
        a.grip_properties = EndEffectorProperties(ui_type=EndEffectorProperties.ELECTRIC_GRIPPER)
        a.grip = EndEffectorState(id=123, calibrated=1, ready=1, error=0, moving=0)
        a.grip_pub = Mock(); a.grip_pub.get_num_connections.return_value = 1
        def receive(msg):
            a.grip.command, a.grip.command_sender = msg.command, msg.sender
            a.grip.command_sequence = msg.sequence
            a.grip.position = __import__('json').loads(msg.args)['position']
            a.times['grip'] = time.monotonic()
        a.grip_pub.publish.side_effect = receive
        with patch('test_grasp_baxter.rospy.get_name', return_value='/offline_test'):
            a.gripper(True)
            self.assertEqual(a.grip.position, 100)
            a.gripper(False)
            self.assertEqual(a.grip.position, 0)
        self.assertEqual(a.grip_pub.publish.call_count, 2)
        for call in a.grip_pub.publish.call_args_list:
            self.assertEqual(call.args[0].command, EndEffectorCommand.CMD_GO)

    def test_preview_cannot_move(self):
        a = arm(); a.execute = False; del a.ensure_limb
        with self.assertRaisesRegex(RuntimeError, '预览'): a.ensure_limb()
        with self.assertRaisesRegex(RuntimeError, '预览'): a.gripper(True)
        a.limb.set_joint_positions.assert_not_called()


if __name__ == '__main__':
    unittest.main()
