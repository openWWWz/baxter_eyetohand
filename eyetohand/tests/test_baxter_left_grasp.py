"""左臂点选控制的接口隔离与运动保护测试，不连接机器人。"""
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import rospy
from baxter_core_msgs.msg import AssemblyState, EndEffectorCommand, EndEffectorProperties, EndEffectorState
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_grasp_baxter_left as left_grasp


LEFT_JOINTS = dict(zip(
    ('left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2'),
    [.1, -.7, .1, 1.2, .1, .3, .1]))
RIGHT_JOINTS = {name.replace('left_', 'right_'): value for name, value in LEFT_JOINTS.items()}


class LeftArmTests(unittest.TestCase):
    def make_arm(self, execute=False):
        with patch.object(left_grasp.rospy, 'Subscriber'), patch.object(left_grasp.rospy, 'Publisher'):
            arm = left_grasp.LeftArm(execute, .1, [0, 0, 0])
        self.addCleanup(arm.cancel.set)
        return arm

    def test_all_arm_and_gripper_interfaces_are_left(self):
        with patch.object(left_grasp.rospy, 'Subscriber') as subscriber, \
                patch.object(left_grasp.rospy, 'Publisher') as publisher:
            arm = left_grasp.LeftArm(True, .1, [0, 0, 0])
        self.assertEqual(arm.ik_name, '/ExternalTools/left/PositionKinematicsNode/IKService')
        self.assertEqual([call.args[0] for call in subscriber.call_args_list], [
            '/robot/limb/left/endpoint_state', '/robot/joint_states', '/robot/state',
            '/robot/end_effector/left_gripper/state', '/robot/end_effector/left_gripper/properties'])
        publisher.assert_called_once_with(
            '/robot/end_effector/left_gripper/command', EndEffectorCommand, queue_size=1)

    def test_preview_does_not_create_command_publisher_or_limb(self):
        with patch.object(left_grasp.rospy, 'Subscriber'), \
                patch.object(left_grasp.rospy, 'Publisher') as publisher, \
                patch('baxter_interface.Limb') as limb:
            arm = left_grasp.LeftArm(False, .1, [0, 0, 0])
            with self.assertRaisesRegex(RuntimeError, '预览'):
                arm.ensure_limb()
            with self.assertRaisesRegex(RuntimeError, '预览'):
                arm.gripper(True)
        publisher.assert_not_called()
        limb.assert_not_called()

    def test_joint_feedback_selects_only_left_from_both_arms(self):
        arm = self.make_arm()
        joints = dict(LEFT_JOINTS, **RIGHT_JOINTS)
        arm.on_state(JointState(name=list(joints), position=list(joints.values())), 'joints')
        self.assertEqual(arm.current_joints(), LEFT_JOINTS)
        other = self.make_arm()
        other.on_state(JointState(name=list(RIGHT_JOINTS), position=list(RIGHT_JOINTS.values())), 'joints')
        self.assertIsNone(other.joints)

    def test_execution_constructs_only_left_sdk_limb(self):
        arm = self.make_arm(True)
        arm.current = Mock()
        arm.wait = Mock()
        arm.require_enabled = Mock()
        with patch('baxter_interface.Limb') as limb:
            arm.ensure_limb()
        limb.assert_called_once_with('left')

    def test_ik_accepts_left_and_rejects_right_mixed_or_invalid(self):
        for kind in (b'\x01', [2], ''):
            response = SimpleNamespace(result_type=kind, isValid=[True], joints=[
                JointState(name=list(LEFT_JOINTS), position=list(LEFT_JOINTS.values()))])
            self.assertEqual(left_grasp.ik_solution(response), LEFT_JOINTS)
        mixed = dict(LEFT_JOINTS)
        mixed['right_w2'] = mixed.pop('left_w2')
        out_of_range = dict(LEFT_JOINTS, left_w1=4)
        for joints in (RIGHT_JOINTS, mixed, out_of_range):
            with self.subTest(joints=joints), self.assertRaises(ValueError):
                left_grasp.ik_solution(SimpleNamespace(result_type=[1], joints=[
                    JointState(name=list(joints), position=list(joints.values()))]))
        with self.assertRaises(ValueError):
            left_grasp.ik_solution(SimpleNamespace(result_type=[0], joints=[]))

    def test_ik_calls_left_service_in_base_frame_with_left_seed(self):
        arm = self.make_arm()
        response = SimpleNamespace(result_type=[1], joints=[
            JointState(name=list(LEFT_JOINTS), position=list(LEFT_JOINTS.values()))])
        service = Mock(return_value=response)
        with patch.object(left_grasp.rospy.Time, 'now', return_value=rospy.Time(1)), \
                patch.object(left_grasp.rospy, 'ServiceProxy', return_value=service) as factory:
            self.assertEqual(arm.call_ik([.4, .2, .2], [0, 0, 0, 1], LEFT_JOINTS), LEFT_JOINTS)
            arm.call_ik([.4, .2, .2], [0, 0, 0, 1], LEFT_JOINTS)
        factory.assert_called_once_with(arm.ik_name, left_grasp.SolvePositionIK, persistent=True)
        request = service.call_args.args[0]
        self.assertEqual(request.pose_stamp[0].header.frame_id, 'base')
        self.assertEqual(dict(zip(request.seed_angles[0].name, request.seed_angles[0].position)), LEFT_JOINTS)

    def test_move_samples_left_orientation_at_p_and_keeps_height_offset(self):
        arm = self.make_arm()
        rotation = Rotation.from_euler('xyz', [.2, -.3, .4])
        arm.current = Mock(return_value=(np.array([.3, .2, .2]), rotation))
        arm.move_to = Mock()
        plan = arm.go_to_target({'surface': [.5, .2, .1], 'quat': [0, 0, 0, 1]}, .08, 0, Mock())
        np.testing.assert_allclose(plan['quat'], rotation.as_quat())
        np.testing.assert_allclose(arm.move_to.call_args.args[0], [.5, .2, .107])

    def test_ik_failure_cannot_send_motion_commands(self):
        arm = self.make_arm(True)
        arm.limb = Mock()
        arm.ensure_limb = Mock()
        arm.current = Mock(return_value=(np.array([.4, .2, .2]), Rotation.identity()))
        arm.current_joints = Mock(return_value=LEFT_JOINTS)
        arm.solve_path = Mock(side_effect=ValueError('IK 无解'))
        with self.assertRaisesRegex(ValueError, 'IK 无解'):
            arm.move_to(np.array([.5, .2, .2]), [0, 0, 0, 1], np.array([.4, .2, .2]))
        arm.limb.set_joint_positions.assert_not_called()
        arm.limb.set_joint_position_speed.assert_not_called()

    def test_cancel_holds_left_joint_positions(self):
        arm = self.make_arm(True)
        arm.limb = Mock()
        arm.current_joints = Mock(return_value=LEFT_JOINTS)
        arm.moving.set()
        arm.stop(announce=False)
        arm.limb.set_joint_positions.assert_called_once_with(LEFT_JOINTS)
        self.assertTrue(arm.cancel.is_set())

    def test_gripper_waits_for_command_acknowledgement(self):
        arm = self.make_arm(True)
        arm.current = Mock()
        arm.wait = Mock()
        arm.require_enabled = Mock()
        arm.times['grip'] = time.monotonic()
        arm.grip_properties = EndEffectorProperties(ui_type=EndEffectorProperties.ELECTRIC_GRIPPER)
        arm.grip = EndEffectorState(id=321, calibrated=1, ready=1, error=0, moving=0)
        arm.grip_pub = Mock()
        arm.grip_pub.get_num_connections.return_value = 1
        def acknowledge(message):
            arm.grip.command = message.command
            arm.grip.command_sender = message.sender
            arm.grip.command_sequence = message.sequence
            arm.grip.position = json.loads(message.args)['position']
            arm.times['grip'] = time.monotonic()
        arm.grip_pub.publish.side_effect = acknowledge
        with patch.object(left_grasp.rospy, 'get_name', return_value='/offline_left_test'):
            arm.gripper(True)
            arm.gripper(False)
        self.assertEqual(arm.grip_pub.publish.call_count, 2)
        self.assertTrue(all(call.args[0].id == 321 for call in arm.grip_pub.publish.call_args_list))

    def test_disabled_robot_or_stale_state_blocks_execution(self):
        arm = self.make_arm(True)
        arm.times = {key: time.monotonic() for key in ('endpoint', 'joints', 'robot')}
        arm.robot = AssemblyState(enabled=False)
        with self.assertRaisesRegex(RuntimeError, '未使能'):
            arm.require_enabled()
        arm.robot.enabled = True
        arm.times['endpoint'] -= 2
        with self.assertRaisesRegex(RuntimeError, '过期'):
            arm.require_enabled()


class LeftCalibrationTests(unittest.TestCase):
    def test_latest_camera_to_base_calibration_is_shared_without_mirroring(self):
        path = Path(__file__).resolve().parents[1] / '8/hand_eye_result.json'
        result = json.loads(path.read_text())
        transform, intrinsics = left_grasp.load_calibration(path)
        np.testing.assert_array_equal(transform, result['T_base_camera'])
        self.assertEqual(intrinsics['width'], 640)

    def test_other_robot_or_transform_convention_is_rejected(self):
        source = Path(__file__).resolve().parents[1] / '8'
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ('hand_eye_result.json', 'intrinsics.json', 'session.json'):
                (folder / name).write_bytes((source / name).read_bytes())
            session = json.loads((folder / 'session.json').read_text())
            session['robot'] = 'OtherRobot'
            (folder / 'session.json').write_text(json.dumps(session))
            with self.assertRaisesRegex(ValueError, 'Baxter'):
                left_grasp.load_calibration(folder / 'hand_eye_result.json')
            session['robot'] = 'Baxter'
            session['arm'] = 'left'
            (folder / 'session.json').write_text(json.dumps(session))
            left_grasp.load_calibration(folder / 'hand_eye_result.json')
            result = json.loads((folder / 'hand_eye_result.json').read_text())
            result['convention'] = 'P_camera = R_camera_base @ P_base + t_camera_base'
            (folder / 'hand_eye_result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, '相机坐标到'):
                left_grasp.load_calibration(folder / 'hand_eye_result.json')

    def test_default_preview_uses_dataset_eight(self):
        with patch.object(sys, 'argv', ['test_grasp_baxter_left.py']):
            args = left_grasp.parse_args()
        self.assertFalse(args.execute)
        self.assertEqual(args.calibration.parent.name, '8')
        self.assertEqual(args.stop_above_mm, 7)
        with patch.object(sys, 'argv', ['test_grasp_baxter_left.py', '--headless-check', '--execute']), \
                patch.object(sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit):
            left_grasp.parse_args()


if __name__ == '__main__':
    unittest.main()
