import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import tetris_env
from tetris_env import ControlAction, FallingPiece, PIECE_NAMES, TetrisEnv


class TetrisEnvTest(unittest.TestCase):
    def test_gymnasium_checker(self) -> None:
        check_env(TetrisEnv(), skip_render_check=True)

    def test_placement_mode_has_rotation_and_x_actions(self) -> None:
        env = gym.make("SimpleTetris-v0")
        obs, info = env.reset(seed=7)
        self.assertEqual(obs.shape, (2, 20, 10))
        self.assertEqual(env.action_space.n, 40)
        valid_action = int(np.flatnonzero(info["action_mask"])[0])
        _, reward, terminated, truncated, info = env.step(valid_action)
        self.assertGreaterEqual(reward, 0.01)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["steps"], 1)

    def test_control_mode_moves_and_rotates_current_piece(self) -> None:
        env = TetrisEnv(action_mode="control")
        env.reset(seed=1)
        initial_x = env.active.x
        env.step(ControlAction.LEFT)
        self.assertEqual(env.active.x, initial_x - 1)
        env.step(ControlAction.ROTATE_CW)
        self.assertIn(env.active.rotation, range(4))

    def test_each_seven_piece_bag_contains_all_pieces(self) -> None:
        env = TetrisEnv()
        env.reset(seed=2)
        names = [env.active.name]
        for _ in range(13):
            env._spawn_piece()
            names.append(env.active.name)
        self.assertEqual(set(names[:7]), set(PIECE_NAMES))
        self.assertEqual(set(names[7:]), set(PIECE_NAMES))

    def test_rotation_kicks_piece_away_from_right_wall(self) -> None:
        env = TetrisEnv(action_mode="control")
        env.reset(seed=3)
        env.active = FallingPiece(name="T", rotation=3, x=8, y=4)
        self.assertTrue(env._try_rotate(1))
        self.assertEqual(env.active.rotation, 0)
        self.assertEqual(env.active.x, 7)


if __name__ == "__main__":
    unittest.main()
