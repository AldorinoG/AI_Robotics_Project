import os
import glob
import importlib.util

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from env_compat import make_car_racing_env

# SETTINGS

TOTAL_TIMESTEPS = 1250000   # steps to train this run (adds on top of existing checkpoint)
SAVE_EVERY      = 25000    # save a checkpoint every N steps
MODEL_NAME      = "racenet_cnn_ppo_legacy"
SAVE_DIR        = "./models"
LOG_DIR         = "./logs"
TENSORBOARD_LOG = LOG_DIR if importlib.util.find_spec("tensorboard") else None

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ENVIRONMENT SETUP

def make_env():
    def _init():
        return make_car_racing_env(legacy_preprocessing=True)
    return _init

def build_env():
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)        # converts (H, W, C) to (C, H, W) for CNN
    env = VecFrameStack(env, n_stack=4) # stack 4 frames so AI sees motion
    return env

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5_000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_print = 0

    def _on_step(self):
        if (
            self.num_timesteps - self.last_print >= self.print_freq
            or self.num_timesteps >= self.total_timesteps
        ):
            self.last_print = self.num_timesteps
            progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            bar_width = 30
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\rProgress: [{bar}] "
                f"{self.num_timesteps:,} / {self.total_timesteps:,} steps "
                f"({progress * 100:5.1f}%)",
                end="",
                flush=True,
            )
        return True

    def _on_training_end(self):
        print()

def main():
    env = build_env()

    # MODEL — auto-resume from latest checkpoint if one exists

    checkpoint_files = glob.glob(f"{SAVE_DIR}/{MODEL_NAME}_*_steps.zip")
    if checkpoint_files:
        latest = max(checkpoint_files, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest}")
        model = PPO.load(latest, env=env, tensorboard_log=TENSORBOARD_LOG)
    else:
        print("No checkpoint found, starting fresh.")
        model = PPO(
            "CnnPolicy",        # CNN processes frames automatically
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
            learning_rate=5e-5,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.03,
        )

    # CALLBACKS

    checkpoint = CheckpointCallback(
        save_freq=SAVE_EVERY,
        save_path=SAVE_DIR,
        name_prefix=MODEL_NAME
    )
    progress = ProgressCallback(TOTAL_TIMESTEPS)
    callbacks = CallbackList([checkpoint, progress])

    # TRAIN

    print("Starting training...")
    print(f"Steps this run: {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints saved to: {SAVE_DIR}")
    if TENSORBOARD_LOG:
        print(f"TensorBoard logs saved to: {TENSORBOARD_LOG}")
    else:
        print("TensorBoard disabled: install it with `pip install tensorboard`")
    print("Press Ctrl+C to stop early - latest checkpoint will be kept\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,  # keeps the total step count accumulating
        )
    except KeyboardInterrupt:
        print("\nTraining stopped early.")

    # SAVE FINAL MODEL

    model.save(f"{SAVE_DIR}/{MODEL_NAME}_final")
    print(f"\nModel saved to {SAVE_DIR}/{MODEL_NAME}_final")

if __name__ == "__main__":
    main()