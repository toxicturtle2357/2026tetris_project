"""A small Gymnasium-compatible Tetris environment for DQN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BOARD_HEIGHT = 20
BOARD_WIDTH = 10


class ControlAction(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROTATE_CW = 3
    ROTATE_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6


PIECES: dict[str, np.ndarray] = {
    "I": np.array([[1, 1, 1, 1]], dtype=np.uint8),
    "O": np.array([[1, 1], [1, 1]], dtype=np.uint8),
    "T": np.array([[0, 1, 0], [1, 1, 1]], dtype=np.uint8),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
    "Z": np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    "J": np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    "L": np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
}
PIECE_NAMES = tuple(PIECES)
LINE_REWARDS = (0.0, 1.0, 3.0, 5.0, 8.0)

# SRS kick tests translated to screen coordinates, where positive y is down.
JLSTZ_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (1, 0): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (1, 2): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (2, 1): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (2, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (3, 2): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (3, 0): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (0, 3): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
}
I_KICKS: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)),
}


@dataclass
class FallingPiece:
    name: str
    rotation: int
    x: int
    y: int


def _trim(shape: np.ndarray) -> np.ndarray:
    rows = np.any(shape, axis=1)
    cols = np.any(shape, axis=0)
    return shape[np.ix_(rows, cols)]


def piece_shape(name: str, rotation: int) -> np.ndarray:
    return _trim(np.rot90(PIECES[name], k=-(rotation % 4)))


class TetrisEnv(gym.Env[np.ndarray, int]):
    """Simplified Tetris: no hold, no preview queue, and no T-spin scoring.

    ``action_mode="placement"`` is intended for early DQN experiments:
    action = rotation * board_width + x. The current piece is rotated and
    hard-dropped at the requested x coordinate.

    ``action_mode="control"`` provides per-frame controls using
    :class:`ControlAction`.
    """

    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 8}

    def __init__(
        self,
        action_mode: str = "placement",
        render_mode: str | None = None,
        max_steps: int = 1000,
    ) -> None:
        super().__init__()
        if action_mode not in {"placement", "control"}:
            raise ValueError("action_mode must be 'placement' or 'control'")
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.action_mode = action_mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(
            4 * BOARD_WIDTH if action_mode == "placement" else len(ControlAction)
        )
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.active: FallingPiece | None = None
        self.steps = 0
        self.total_lines = 0
        self._bag: list[str] = []
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None
        self._font: Any | None = None
        self._small_font: Any | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.board.fill(0)
        self.steps = 0
        self.total_lines = 0
        self._bag.clear()
        self._spawn_piece()
        return self._get_obs(), self._get_info(lines_cleared=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.active is None:
            raise RuntimeError("Call reset() before step(), or reset after termination.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        if self.action_mode == "placement":
            reward, terminated, lines = self._step_placement(int(action))
        else:
            reward, terminated, lines = self._step_control(ControlAction(int(action)))

        truncated = self.steps >= self.max_steps and not terminated
        info = self._get_info(lines_cleared=lines)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def _step_placement(self, action: int) -> tuple[float, bool, int]:
        assert self.active is not None
        rotation, x = divmod(action, BOARD_WIDTH)
        shape = piece_shape(self.active.name, rotation)
        if x + shape.shape[1] > BOARD_WIDTH or self._collides(shape, x, 0):
            return -0.25, False, 0

        self.active.rotation = rotation
        self.active.x = x
        self.active.y = 0
        while not self._collides(shape, self.active.x, self.active.y + 1):
            self.active.y += 1
        return self._lock_active()

    def _step_control(self, action: ControlAction) -> tuple[float, bool, int]:
        assert self.active is not None
        if action == ControlAction.LEFT:
            self._try_move(dx=-1)
            return 0.0, False, 0
        elif action == ControlAction.RIGHT:
            self._try_move(dx=1)
            return 0.0, False, 0
        elif action == ControlAction.ROTATE_CW:
            self._try_rotate(1)
            return 0.0, False, 0
        elif action == ControlAction.ROTATE_CCW:
            self._try_rotate(-1)
            return 0.0, False, 0
        elif action == ControlAction.HARD_DROP:
            shape = self._active_shape()
            while not self._collides(shape, self.active.x, self.active.y + 1):
                self.active.y += 1
            return self._lock_active()
        elif action == ControlAction.SOFT_DROP:
            if self._try_move(dy=1):
                return 0.001, False, 0
            return self._lock_active()

        if action == ControlAction.NOOP and not self._try_move(dy=1):
            return self._lock_active()
        return 0.0, False, 0

    def _try_move(self, dx: int = 0, dy: int = 0) -> bool:
        assert self.active is not None
        shape = self._active_shape()
        x = self.active.x + dx
        y = self.active.y + dy
        if self._collides(shape, x, y):
            return False
        self.active.x = x
        self.active.y = y
        return True

    def _try_rotate(self, delta: int) -> bool:
        assert self.active is not None
        old_rotation = self.active.rotation
        rotation = (old_rotation + delta) % 4
        shape = piece_shape(self.active.name, rotation)
        if self.active.name == "O":
            tests = ((0, 0),)
        elif self.active.name == "I":
            tests = I_KICKS[(old_rotation, rotation)]
        else:
            tests = JLSTZ_KICKS[(old_rotation, rotation)]
        for dx, dy in tests:
            x = self.active.x + dx
            y = self.active.y + dy
            if not self._collides(shape, x, y):
                self.active.rotation = rotation
                self.active.x = x
                self.active.y = y
                return True
        return False

    def _active_shape(self) -> np.ndarray:
        assert self.active is not None
        return piece_shape(self.active.name, self.active.rotation)

    def _collides(self, shape: np.ndarray, x: int, y: int) -> bool:
        height, width = shape.shape
        if x < 0 or x + width > BOARD_WIDTH or y < 0 or y + height > BOARD_HEIGHT:
            return True
        return bool(np.any(self.board[y : y + height, x : x + width] & shape))

    def _spawn_piece(self) -> bool:
        if not self._bag:
            self._bag = list(PIECE_NAMES)
            self.np_random.shuffle(self._bag)
        name = self._bag.pop()
        shape = piece_shape(name, 0)
        self.active = FallingPiece(
            name=name, rotation=0, x=(BOARD_WIDTH - shape.shape[1]) // 2, y=0
        )
        return not self._collides(shape, self.active.x, self.active.y)

    def _lock_active(self) -> tuple[float, bool, int]:
        assert self.active is not None
        shape = self._active_shape()
        height, width = shape.shape
        self.board[
            self.active.y : self.active.y + height,
            self.active.x : self.active.x + width,
        ] |= shape
        full_rows = np.all(self.board == 1, axis=1)
        lines = int(np.count_nonzero(full_rows))
        if lines:
            remaining = self.board[~full_rows]
            self.board = np.vstack(
                [np.zeros((lines, BOARD_WIDTH), dtype=np.uint8), remaining]
            )
        self.total_lines += lines
        terminated = not self._spawn_piece()
        reward = LINE_REWARDS[lines] + 0.01
        if terminated:
            reward -= 1.0
        return reward, terminated, lines

    def _get_obs(self) -> np.ndarray:
        active_layer = np.zeros_like(self.board)
        if self.active is not None:
            shape = self._active_shape()
            height, width = shape.shape
            active_layer[
                self.active.y : self.active.y + height,
                self.active.x : self.active.x + width,
            ] = shape
        return np.stack([self.board, active_layer]).astype(np.uint8, copy=False)

    def _get_info(self, lines_cleared: int) -> dict[str, Any]:
        info: dict[str, Any] = {
            "lines_cleared": lines_cleared,
            "total_lines": self.total_lines,
            "steps": self.steps,
        }
        if self.active is not None:
            info["current_piece"] = self.active.name
        if self.action_mode == "placement":
            info["action_mask"] = self.valid_action_mask()
        return info

    def valid_action_mask(self) -> np.ndarray:
        if self.action_mode != "placement":
            raise RuntimeError("Action masks are only defined for placement mode.")
        assert self.active is not None
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            rotation, x = divmod(action, BOARD_WIDTH)
            shape = piece_shape(self.active.name, rotation)
            mask[action] = x + shape.shape[1] <= BOARD_WIDTH and not self._collides(
                shape, x, 0
            )
        return mask

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            return None
        obs = self._get_obs()
        display = np.maximum(obs[0], obs[1] * 2)
        if self.render_mode == "rgb_array":
            colors = np.array([[18, 18, 24], [74, 151, 255], [255, 195, 70]], dtype=np.uint8)
            return colors[display]
        if self.render_mode == "human":
            self._render_pygame(display)
            return None
        text = "\n".join(
            "|" + "".join("  " if cell == 0 else "[]" if cell == 1 else "<>" for cell in row) + "|"
            for row in display
        )
        text = text + "\n+" + "--" * BOARD_WIDTH + "+"
        return text

    def _render_pygame(self, display: np.ndarray) -> None:
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise ImportError(
                    "pygame is required for render_mode='human'. "
                    "Install it with: pip install pygame"
                ) from exc
            self._pygame = pygame
            pygame.init()
            pygame.display.set_caption("Simple Tetris - Gymnasium Environment")
            self._screen = pygame.display.set_mode((500, 660))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("arial", 25, bold=True)
            self._small_font = pygame.font.SysFont("arial", 18)

        pygame = self._pygame
        assert self._screen is not None
        assert self._clock is not None
        assert self._font is not None
        assert self._small_font is not None

        cell = 30
        board_x, board_y = 20, 30
        board_width = BOARD_WIDTH * cell
        board_height = BOARD_HEIGHT * cell
        background = (13, 16, 28)
        panel = (22, 27, 43)
        grid = (44, 50, 67)
        fixed = (43, 128, 255)
        active = (251, 191, 36)
        white = (238, 242, 255)
        muted = (154, 164, 187)

        self._screen.fill(background)
        pygame.draw.rect(
            self._screen,
            panel,
            (board_x, board_y, board_width, board_height),
            border_radius=5,
        )
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                rect = pygame.Rect(
                    board_x + x * cell + 1, board_y + y * cell + 1, cell - 2, cell - 2
                )
                if display[y, x] == 1:
                    pygame.draw.rect(self._screen, fixed, rect, border_radius=5)
                    pygame.draw.line(
                        self._screen, (104, 173, 255), rect.topleft, rect.topright, 2
                    )
                elif display[y, x] == 2:
                    pygame.draw.rect(self._screen, active, rect, border_radius=5)
                    pygame.draw.line(
                        self._screen, (255, 224, 126), rect.topleft, rect.topright, 2
                    )
                else:
                    pygame.draw.rect(self._screen, grid, rect, 1, border_radius=4)

        panel_x = board_x + board_width + 28
        title = self._font.render("TETRIS", True, white)
        self._screen.blit(title, (panel_x, 48))
        lines_label = self._small_font.render("LINES", True, muted)
        lines_value = self._font.render(str(self.total_lines), True, white)
        self._screen.blit(lines_label, (panel_x, 130))
        self._screen.blit(lines_value, (panel_x, 158))
        piece_label = self._small_font.render("PIECE", True, muted)
        piece_value = self._font.render(
            self.active.name if self.active else "-", True, active
        )
        self._screen.blit(piece_label, (panel_x, 222))
        self._screen.blit(piece_value, (panel_x, 250))

        if self.action_mode == "control":
            controls = ("KEYS", "Left/Right", "Move", "Up / Z", "Rotate", "Space", "Drop")
            y = 350
            for index, line in enumerate(controls):
                color = muted if index % 2 else white
                self._screen.blit(self._small_font.render(line, True, color), (panel_x, y))
                y += 26

        pygame.draw.rect(
            self._screen,
            (82, 90, 114),
            (board_x - 2, board_y - 2, board_width + 4, board_height + 4),
            2,
            border_radius=7,
        )
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None


def register_tetris_env() -> None:
    if "SimpleTetris-v0" not in gym.registry:
        gym.register(
            id="SimpleTetris-v0",
            entry_point="tetris_env:TetrisEnv",
            max_episode_steps=1000,
        )


register_tetris_env()
