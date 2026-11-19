"""采集文件格式与动作回放的离线验证，不连接或驱动 Baxter。"""

import argparse
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accurate_eye_to_hand import parse_args
from accurate_eye_to_hand import load_views
from record_baxter import (Dataset, Receiver, arm_joint_names, joint_values, load_actions,
                           next_output_dir, replay, validate_action)


TARGET = dict(zip(arm_joint_names('right'), [0.2, -0.7, 0.1, 1.2, 0.2, 0.4, 0.3]))
INTR = {
    'width': 2, 'height': 1, 'model': 'plumb_bob',
    'camera_matrix': [[400, 0, 1], [0, 400, 0], [0, 0, 1]],
    'dist_coeffs': [0] * 5,
}


class FakeReceiver:
    def __init__(self, actual):
        self.condition = threading.Condition()
        self.joint_seq = 1
        self.joint_state = JointState(name=list(actual), position=list(actual.values()))
        self.captured = 0

    def wait_for(self, predicate, timeout, message):
        self.joint_seq += 1
        if not predicate():
            raise RuntimeError(message)

    def capture(self, intrinsics):
        self.captured += 1
        return np.zeros((1, 2, 3), np.uint8), [1, 2, 3, 0, 0, 0], joint_values(self.joint_state, 'right'), {}


class FakeLimb:
    def __init__(self):
        self.moves = []
        self.speeds = []
        self.holds = []

    def set_joint_position_speed(self, speed):
        self.speeds.append(speed)

    def move_to_joint_positions(self, positions, **kwargs):
        self.moves.append((positions, kwargs))
        # 模拟 Baxter SDK：即使超时也直接返回。

    def joint_angles(self):
        return TARGET

    def set_joint_positions(self, positions):
        self.holds.append(positions)


class RecordBaxterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_dataset(self, name='1'):
        return Dataset(self.root / name, INTR, 'right', '/baxter_camera/color/image_raw')

    def test_save_picture_action_and_pose_are_matched(self):
        dataset = self.make_dataset()
        dataset.save(np.zeros((1, 2, 3), np.uint8), [1, 2, 3, 0, 0, 0], TARGET, {'image_frame_id': 'camera'})
        dataset.save(np.zeros((1, 2, 3), np.uint8), [2, 3, 4, 0, 0, 0], TARGET, {})
        self.assertTrue((dataset.folder / 'picture/0.png').is_file())
        self.assertTrue((dataset.folder / 'picture/1.png').is_file())
        self.assertEqual((dataset.folder / 'record.txt').read_text().splitlines(),
                         ['1,2,3,0,0,0', '2,3,4,0,0,0'])
        source, values = load_actions(dataset.folder / 'action', 'right')[0]
        self.assertEqual(source.name, '0.json')
        self.assertEqual(values, TARGET)
        with patch.object(sys, 'argv', ['accurate_eye_to_hand.py', '--dataset', str(dataset.folder)]):
            self.assertEqual(parse_args().image_dir, dataset.folder / 'picture')
        self.assertEqual(next_output_dir(self.root), self.root / '2')

    def test_replay_saves_new_measurement_after_arrival(self):
        original = self.make_dataset()
        original.save(np.zeros((1, 2, 3), np.uint8), [1, 2, 3, 0, 0, 0], TARGET, {})
        replay_dataset = self.make_dataset('2')
        limb = FakeLimb()
        receiver = FakeReceiver(TARGET)
        args = argparse.Namespace(arm='right', speed=0.15, move_timeout=10.0, settle_seconds=0.0)
        replay(load_actions(original.folder / 'action/0.json', 'right'), limb, receiver,
               replay_dataset, INTR, args, threading.Event())
        self.assertEqual(replay_dataset.count, 1)
        self.assertEqual(receiver.captured, 1)
        self.assertEqual(limb.moves[0][1]['timeout'], 10.0)
        self.assertEqual(limb.speeds, [0.15, 0.3])
        data = json.loads((replay_dataset.folder / 'action/0.json').read_text())
        self.assertEqual(data['joint_positions_rad'], TARGET)
        self.assertTrue(data['replayed_from'].endswith('/1/action/0.json'))

    def test_replay_refuses_to_photograph_when_sdk_returns_before_arrival(self):
        original = self.make_dataset()
        original.save(np.zeros((1, 2, 3), np.uint8), [1, 2, 3, 0, 0, 0], TARGET, {})
        dataset = self.make_dataset('2')
        incorrect = dict(TARGET, right_s0=TARGET['right_s0'] + 0.1)
        receiver, limb = FakeReceiver(incorrect), FakeLimb()
        args = argparse.Namespace(arm='right', speed=0.15, move_timeout=10, settle_seconds=0)
        with self.assertRaisesRegex(RuntimeError, '未到位'):
            replay(load_actions(original.folder / 'action', 'right'), limb, receiver, dataset,
                   INTR, args, threading.Event())
        self.assertEqual(dataset.count, 0)
        self.assertEqual(receiver.captured, 0)
        self.assertEqual(limb.speeds, [0.15, 0.3])
        self.assertEqual(len(limb.holds), 1)

    def test_wrong_arm_and_limits_are_rejected(self):
        data = {'format': 'baxter_joint_sample_v1', 'arm': 'right', 'joint_positions_rad': TARGET}
        with self.assertRaises(ValueError):
            validate_action(data, 'left')
        with self.assertRaises(ValueError):
            validate_action(dict(data, joint_positions_rad=dict(TARGET, right_e1=3.0)), 'right')
        gripper_only = JointState(name=['right_gripper_l_finger_joint'], position=[0.1])
        with self.assertRaises(ValueError):
            joint_values(gripper_only, 'right')
        receiver = object.__new__(Receiver)
        receiver.arm, receiver.joint_seq = 'right', 0
        receiver.condition = threading.Condition()
        receiver.on_joint(gripper_only)
        self.assertEqual(receiver.joint_seq, 0)
        receiver.on_joint(JointState(name=list(TARGET), position=list(TARGET.values())))
        self.assertEqual(receiver.joint_seq, 1)

    def test_excluding_image_preserves_original_pose_pairing(self):
        pictures = self.root / 'picture'
        pictures.mkdir()
        for index in range(4):
            (pictures / f'{index}.png').touch()
        records = self.root / 'record.txt'
        records.write_text(''.join(f'{index},0,0,0,0,0\n' for index in range(4)))
        args = argparse.Namespace(image_dir=pictures, record_file=records, euler_order='xyz',
                                  cols=3, rows=2, square_size=.015, use_sb_first=True,
                                  exclude_images=[1])
        corners = np.zeros((6, 1, 2), dtype=np.float32)
        with patch('accurate_eye_to_hand.detect_chessboard', return_value=(True, (640, 480), corners)):
            views, _ = load_views(args)
        self.assertEqual([view.image_id for view in views], [0, 2, 3])
        self.assertEqual([view.T_base_gripper[0, 3] for view in views], [0, 2, 3])


if __name__ == '__main__':
    unittest.main()
