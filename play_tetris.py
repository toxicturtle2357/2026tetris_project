"""Open the environment as a small playable Tetris window."""

from __future__ import annotations

import time

import pygame

from tetris_env import ControlAction, TetrisEnv


DROP_INTERVAL_SECONDS = 0.45


def main() -> None:
    env = TetrisEnv(action_mode="control", render_mode="human", max_steps=100_000)
    env.reset()
    env.render()
    running = True
    finished = False
    last_drop = time.monotonic()

    while running:
        action: ControlAction | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif finished and event.key == pygame.K_r:
                    env.reset()
                    env.render()
                    finished = False
                    last_drop = time.monotonic()
                elif not finished:
                    action = {
                        pygame.K_LEFT: ControlAction.LEFT,
                        pygame.K_RIGHT: ControlAction.RIGHT,
                        pygame.K_UP: ControlAction.ROTATE_CW,
                        pygame.K_z: ControlAction.ROTATE_CCW,
                        pygame.K_DOWN: ControlAction.SOFT_DROP,
                        pygame.K_SPACE: ControlAction.HARD_DROP,
                    }.get(event.key)

        now = time.monotonic()
        if not finished and action is not None:
            _, _, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            last_drop = now
        elif not finished and now - last_drop >= DROP_INTERVAL_SECONDS:
            _, _, terminated, truncated, _ = env.step(ControlAction.NOOP)
            finished = terminated or truncated
            last_drop = now

        if finished:
            _draw_game_over(env)
        pygame.time.wait(10)

    env.close()


def _draw_game_over(env: TetrisEnv) -> None:
    env.render()
    pygame = env._pygame
    screen = env._screen
    if pygame is None or screen is None:
        return
    shade = pygame.Surface((300, 110), pygame.SRCALPHA)
    shade.fill((5, 8, 16, 220))
    screen.blit(shade, (20, 274))
    title = env._font.render("GAME OVER", True, (250, 90, 100))
    retry = env._small_font.render("Press R to restart", True, (238, 242, 255))
    screen.blit(title, (86, 294))
    screen.blit(retry, (86, 340))
    pygame.display.flip()


if __name__ == "__main__":
    main()
