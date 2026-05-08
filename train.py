import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

#SETTINGS

TOTAL_TIMESTEPS = 500_000   # increase to 1_000_000 for better results
SAVE_EVERY      = 50_000    # saves a checkpoint every 50k steps
MODEL_NAME      = "racenet_cnn_ppo"
SAVE_DIR        = "./models"
LOG_DIR         = "./logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

#ENVIRONMENT SETUP

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", continuous=True)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)    # converts (H, W, C) → (C, H, W) for CNN
env = VecFrameStack(env, n_stack=4)  # stack 4 frames so AI sees motion

#MODEL

model = PPO(
    "CnnPolicy",        # CNN processes frames automatically
    env,
    verbose=1,          # prints training progress
    tensorboard_log=LOG_DIR,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
)

#CALLBACKS

# Saves model every SAVE_EVERY steps so you don't lose progress
checkpoint = CheckpointCallback(
    save_freq=SAVE_EVERY,
    save_path=SAVE_DIR,
    name_prefix=MODEL_NAME
)

#TRAIN

print("Starting training...")
print(f"Total steps: {TOTAL_TIMESTEPS:,}")
print(f"Checkpoints saved to: {SAVE_DIR}")
print("Press Ctrl+C to stop early — latest checkpoint will be kept\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("\nTraining stopped early.")

#SAVE FINAL MODEL

model.save(f"{SAVE_DIR}/{MODEL_NAME}_final")
print(f"\nModel saved to {SAVE_DIR}/{MODEL_NAME}_final")