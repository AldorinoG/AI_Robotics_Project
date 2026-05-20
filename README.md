# Apex AI — Human vs Machine Racing

Apex AI is a reinforcement learning project built on OpenAI Gymnasium's CarRacing-v3 environment. A CNN (Convolutional Neural Network) processes raw 96×96 pixel camera frames to extract visual features, which feed into a PPO (Proximal Policy Optimisation) agent that learns to steer, accelerate, and brake around a procedurally generated race track. The goal is for the AI to beat lap times and coin scores set by human players through a manual driving mode.

---

## Group Information

**Team 2**

| Name | Student ID |
|---|---|
| Aldorino Gladwyn | |
| Maksim Lovchev | |
| Mikhail Lazarev | |
| Hong Nam Hoang | |

---

## How It Works

The AI pipeline runs as follows on every simulation step:

```
Raw RGB Frame (96x96x3)
        |
        v
Greyscale Preprocessing
  Converts to single-channel (96x96x1)
        |
        v
CNN (Convolutional Neural Network)
  Extracts spatial features from the frame
  4 frames stacked so the agent perceives motion
        |
        v
PPO Agent (Proximal Policy Optimisation)
  Decides: steer left/right, accelerate, brake
        |
        v
CarRacing-v3 Environment
```

### Team Strategy — Staged Rubric

| Grade | Requirement | Implementation |
|---|---|---|
| **Pass** | Working PPO agent that can drive on the track | PPO trained on CarRacing-v3 using `stable-baselines3` |
| **Credit** | CNN processes frames and feeds into PPO during live simulation | `CnnPolicy` + `VecFrameStack` (4 frames) running in real time |
| **Distinction** | PPO agent completes at least 50% of the track consistently | Achievable at 1M training steps with reward shaping and tuned hyperparameters |
| **HD** | AI navigates the full track and beats human recorded lap time | Target at 2M+ steps — leaderboard compares AI vs human times directly |

---

## Dependencies

Python 3.11 is required. Install all dependencies with:

```bash
pip install stable-baselines3 gymnasium[box2d] pygame numpy
```

Optional — enables training loss charts:

```bash
pip install tensorboard
```

For GPU-accelerated training (NVIDIA only), reinstall PyTorch with CUDA support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (check with `nvidia-smi`).

---

## Project Structure

```
AI_Robotics_Project/
├── train.py          # Train the PPO agent
├── run_car.py        # Launch the game (manual or AI drive)
├── env_compat.py     # Gymnasium environment wrappers and reward shaping
├── models/           # Saved model checkpoints (created automatically)
├── logs/             # TensorBoard training logs (created automatically)
└── leaderboard.json  # Persistent leaderboard per map and mode (created automatically)
```

---

## How to Run

### 1. Train the AI

```bash
py -3.11 train.py
```

- Trains for 1,250,000 steps by default
- Checkpoints are saved every 25,000 steps to `./models/`
- Press `Ctrl+C` to stop early — the latest checkpoint is kept automatically
- Run again at any time to **resume training from the last checkpoint automatically**
- Expected output:
  ```
  Resuming from checkpoint: ./models/racenet_cnn_ppo_legacy_250000_steps.zip
  Starting training...
  Steps this run: 250,000
  Progress: [####--------] 150,000 / 250,000 steps ( 60.0%)
  ```

### 2. Run the Game

```bash
py -3.11 run_car.py
```

A launcher window will appear with the following options:

**Drive Manually**
- Enter your name, choose a map and game mode (Collect Coins or Best Time)
- Controls: `W` = accelerate, `S` = brake, `A` = steer left, `D` = steer right, `Q` = quit
- **Collect Coins mode**: collect as many track tiles as possible within 30 seconds
- **Best Time mode**: complete the full lap as fast as possible — lap ends when you cross the start/finish line
- Your score is saved to the leaderboard automatically, separated by map and mode

**Let AI Drive**
- Requires a trained model in `./models/` — run `train.py` first
- The AI drives autonomously and its score is compared against the human leaderboard for that map and mode

### 3. Maps

There are 3 maps available, each with a different procedurally generated track layout:

| Map | Seed |
|---|---|
| Map 1 | 21 |
| Map 2 | 84 |
| Map 3 | 126 |

### 4. Leaderboard

Scores are stored separately per **map** and per **game mode**. Selecting a map and mode in the launcher shows the current top 3 for that combination. Each player only keeps their personal best score.

---

## Game Window

The game renders a full-screen camera view of the car with a **minimap overlay** in the bottom-right corner showing:
- The full track layout
- Collected tiles (green) vs uncollected tiles (yellow)
- The start/finish line (white dot marked S)
- The car's current position (red dot)

Use the **mouse scroll wheel** to zoom the camera in or out.

---

## Lap Completion

A lap is considered complete when the car **crosses back over the start/finish line** (tile 0) after having driven at least 50% of the track. This applies to both human and AI modes.

- **Best Time mode**: the clock stops when the lap is complete
- **Collect Coins mode**: the round ends when the 30-second timer expires, regardless of lap progress

---

## Expected Output

After training completes, the AI should be able to navigate the track and collect coins or complete laps. Performance improves significantly with more training steps:

| Steps | Expected Behaviour |
|---|---|
| 250,000 | Drives forward, frequently goes off-track |
| 500,000 | Completes easier maps occasionally |
| 1,000,000 | Completes most maps with some mistakes |
| 2,000,000+ | Consistent, clean laps — HD territory |

---

## Resources & References

### Core Libraries

- **Stable-Baselines3** — PPO implementation and vectorised environment utilities
  https://stable-baselines3.readthedocs.io

- **Gymnasium CarRacing-v3** — the race track simulation environment
  https://gymnasium.farama.org/environments/box2d/car_racing/

- **PyTorch** — underlying deep learning framework used by Stable-Baselines3
  https://pytorch.org

- **NumPy** — numerical operations and array handling
  https://numpy.org

### Tools

- **Pygame** — game window rendering and human input handling
  https://www.pygame.org

- **TensorBoard** — optional training loss visualisation
  https://www.tensorflow.org/tensorboard


---