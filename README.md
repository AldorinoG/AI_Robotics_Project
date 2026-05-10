# AI Robotics Car Racing Project

This project trains and runs a PPO self-driving agent for Gymnasium `CarRacing-v3`.

## Files

- `train.py` trains the racing model and saves checkpoints in `models/`.
- `run_car.py` opens the launcher for human driving or AI driving.
- `env_compat.py` adapts Gymnasium `CarRacing-v3` for the installed Stable-Baselines3 version and applies the current observation/action preprocessing.
- `leaderboard.json` stores lap times.

## Train

```bash
source venv/bin/activate
python train.py
```

Training checkpoints are saved in `models/`, and TensorBoard logs are saved in `logs/` when TensorBoard is installed.

## Run

```bash
source venv/bin/activate
python run_car.py
```

