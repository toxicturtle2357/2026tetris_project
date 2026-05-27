# Simple Tetris Gymnasium Environment

DQN 학습을 위한 간단한 테트리스 환경입니다. `Gymnasium`의 커스텀 환경 API를
따르며, `hold`, 다음 블록 미리보기, T-spin 판정/추가 점수는 포함하지 않습니다.
피스는 표준적인 `7-bag` 방식으로 생성되고, 직접 조작 모드의 회전에는
SRS 기반 wall kick이 적용됩니다.
관측값은 **쌓인 블록**과 **현재 조작 중인 블록**의 두 채널뿐입니다.

## 설치

```powershell
pip install -r requirements.txt
```

## DQN에 권장하는 액션: placement

`placement`가 기본 모드입니다. 액션 하나가 현재 블록의 회전과 놓을 x 좌표를
결정한 뒤 하드 드롭합니다. 버튼을 여러 프레임 동안 눌러야 하는 문제를 피하므로
첫 DQN 실험에 적합합니다.

```python
import gymnasium as gym
import tetris_env  # SimpleTetris-v0 등록

env = gym.make("SimpleTetris-v0", action_mode="placement")
obs, info = env.reset(seed=42)

# action = rotation * 10 + x
# 회전 1번, x=4 위치에 현재 블록을 놓음
obs, reward, terminated, truncated, info = env.step(1 * 10 + 4)
```

액션 공간은 `Discrete(40)`입니다.

| 값 | 의미 |
| --- | --- |
| `action // 10` | 시계방향 회전 수 (`0`~`3`) |
| `action % 10` | 블록의 왼쪽 끝 x 좌표 (`0`~`9`) |

블록 너비 때문에 놓을 수 없는 x 좌표는 `info["action_mask"]`가 `False`이며,
선택 시 `-0.25` 보상을 받고 현재 블록은 그대로 유지됩니다.

## 직접 이동 액션: control

버튼 단위 동작이 필요하면 다음 모드를 사용합니다.

```python
from tetris_env import ControlAction, TetrisEnv

env = TetrisEnv(action_mode="control", render_mode="ansi")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(ControlAction.LEFT)
print(env.render())
```

| 액션 | 동작 |
| --- | --- |
| `0` | 자연 낙하 한 칸 |
| `1`, `2` | 왼쪽, 오른쪽 이동 |
| `3`, `4` | 시계방향, 반시계방향 회전 |
| `5` | 소프트 드롭 |
| `6` | 하드 드롭 |

## 관측값과 보상

- 관측 공간: `Box(0, 1, (2, 20, 10), uint8)`
- 채널 `0`: 이미 고정된 블록
- 채널 `1`: 현재 조작할 블록
- 줄 제거 보상: 1줄 `1`, 2줄 `3`, 3줄 `5`, 4줄 `8`
- 블록을 정상적으로 놓으면 `+0.01`, 게임 오버 시 `-1`

`info`에는 학습 입력이 아닌 기록용으로 `lines_cleared`, `total_lines`,
`steps`, `current_piece`가 들어갑니다. 다음에 나올 블록 정보는 없습니다.

## DQN 코드에 연결하기

다른 학습 코드에서는 환경 파일을 import해 등록한 다음 일반 Gymnasium 환경처럼
생성하면 됩니다.

```python
import gymnasium as gym
from stable_baselines3 import DQN
import tetris_env

env = gym.make("SimpleTetris-v0", action_mode="placement")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)
```

`placement` 모드는 `Discrete(40)`이므로 Stable-Baselines3의 기본 DQN에 바로
연결할 수 있습니다. `info["action_mask"]`는 배치할 수 없는 x 위치를 제공하지만
기본 DQN이 자동으로 사용하지는 않으므로, 처음 적용할 때는 그대로 학습시키거나
마스크를 지원하는 별도 정책 처리를 추가해야 합니다.

## 환경만 실행해 보기

DQN 학습을 시작하지 않고, 환경이 만들어지고 액션을 받아 보드가 바뀌는지만
확인하려면 아래 파일을 실행합니다. 유효한 `회전 + x 좌표` 액션을 15번
무작위로 선택해 텍스트 보드를 출력합니다.

```powershell
python run_environment.py
```

## 게임 창으로 직접 확인하기

같은 Gymnasium 환경을 그래픽 게임 창으로 열고 직접 움직일 수 있습니다. 이는
환경 확인용이며 DQN 학습은 실행하지 않습니다.

```powershell
python play_tetris.py
```

| 키 | 동작 |
| --- | --- |
| `Left`, `Right` | x 좌표 이동 |
| `Up`, `Z` | 시계/반시계 방향 회전 |
| `Down` | 소프트 드롭 |
| `Space` | 하드 드롭 |
| `R` | 게임 오버 후 재시작 |
| `Esc` | 종료 |

`render_mode="human"`을 사용하면 환경 자체가 `pygame` 창으로 보드를
그립니다. 학습에는 기존 `placement` 모드와 숫자 관측값을 그대로 사용할 수
있습니다.

## DQN 학습 실행

```powershell
python train_dqn.py
```

기본 예시는 `stable-baselines3`의 `DQN("MlpPolicy", ...)`로 30만 스텝을
학습하고 `tetris_dqn.zip`을 생성합니다.

## 확인

```powershell
python -m unittest -v test_tetris_env.py
```

환경 구현은 Gymnasium 공식 문서의 커스텀 환경 형태(`reset`, `step`,
`action_space`, `observation_space`)를 따릅니다.
