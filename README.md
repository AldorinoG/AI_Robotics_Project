# Apex AI — Human vs Machine Racing

Apex AI is a reinforcement learning project built on OpenAI Gymnasium's CarRacing-v3 environment. A CNN (Convolutional Neural Network) processes raw 96×96 pixel camera frames to extract visual features, which feed into a PPO (Proximal Policy Optimisation) agent that learns to steer, accelerate, and brake around a procedurally generated race track. The goal is for the AI to beat lap times and coin scores set by human players through a manual driving mode.

---

## Group Information

**Team 2**

| Name                 | Student ID |
|----------------------|------------|
|  Aldorino Gladwyn    | 14468944   |
|  Maksim Lovchev      |            |
|  Mikhail Lazarev     |            |
|  Hong Nam Hoang      |            |
|  

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
|-------|-------------|----------------|
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
└── leaderboard.json  # Persistent leaderboard (created automatically)
```

---

## How to Run

### 1. Train the AI

```bash
py -3.11 train.py
```

- Trains for 1,000,000 steps by default
- Checkpoints are saved every 50,000 steps to `./models/`
- Press `Ctrl+C` to stop early — progress is saved automatically
- Run again at any time to resume training from the last save
- Expected output:
  ```
  No saved model found — starting from scratch.
  Starting training...
  Target steps: 1,000,000
  Progress: [####--------] 150,000 / 1,000,000 steps ( 15.0%)
  ```

### 2. Run the Game

```bash
py -3.11 run_car.py
```

A launcher window will appear with the following options:

**Drive Manually**
- Enter your name, choose a map and game mode (Collect Coins or Best Time)
- Controls: `W` = accelerate, `S` = brake, `A` = steer left, `D` = steer right, `Q` = quit
- Coins mode: collect as many track tiles as possible within 30 seconds
- Best Time mode: complete the full lap as fast as possible
- Your score is saved to the leaderboard automatically

**Let AI Drive**
- Requires a trained model in `./models/` — run `train.py` first
- The AI drives autonomously and its score is compared against the human leaderboard
- Press `M` in the viewer window to toggle between map view and camera view

---

## Expected Output

After training completes, the AI should be able to navigate the track and collect coins or complete laps. Performance improves significantly with more training steps:

| Steps     | Expected Behaviour                          |
|-----------|---------------------------------------------|
| 250,000   | Drives forward, frequently goes off-track   |
| 500,000   | Completes easier maps occasionally          |
| 1,000,000 | Completes most maps with some mistakes      |
| 2,000,000+| Consistent, clean laps — HD territory       |

---

## Resources & References

<!-- Add your references here -->

---