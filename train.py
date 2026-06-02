import os
import glob
import importlib.util

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from collections import deque
import pygame

from env_compat import make_car_racing_env

# ════════════════════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# ════════════════════════════════════════════════════════════════════════════════

TOTAL_TIMESTEPS = 1_200_000       # steps to train this run (accumulates on top of checkpoint)
SAVE_EVERY      = 25_000        # checkpoint interval
MODEL_NAME      = "updated_ppo_car_racing"
SAVE_DIR        = "./models"
LOG_DIR         = "./logs"
TENSORBOARD_LOG = LOG_DIR if importlib.util.find_spec("tensorboard") else None
NUM_ENVS        = 8
USE_SUBPROC     = True          # set False to debug with DummyVecEnv

# ════════════════════════════════════════════════════════════════════════════════
# REWARD & TERMINATION  ← all tuning lives here, nowhere else
# ════════════════════════════════════════════════════════════════════════════════

REWARD_SHAPING          = True

# ── Termination ──────────────────────────────────────────────────────────────
# Car is terminated (and penalised) if it makes zero tile progress for this
# many consecutive steps.  25 steps is tight enough to kill donuts quickly
# while still allowing the car to navigate a sharp corner.
TERMINATE_NO_PROGRESS   = True
NO_PROGRESS_STEPS_LIMIT = 30        # was 25 — increased to allow slow navigation of sharp turns
TERMINATION_PENALTY     = 400.0       # was 1.0 — must hurt more than circling rewards

# ── Per-step penalties ────────────────────────────────────────────────────────
NO_PROGRESS_PENALTY     = 0.00       # was 0.3 — slightly reduced to keep it moving while searching
OFF_TRACK_PENALTY       = 2.0       # immediate penalty for being outside the track lines
TIME_PENALTY_PER_STEP   = 0.001     # small constant to discourage dawdling

# ── Bonuses ───────────────────────────────────────────────────────────────────
COMPLETION_BONUS        = 500.0      # was 5.0 — increased reward for winning

# ── Reward weights ────────────────────────────────────────────────────────────
PROGRESS_REWARD_PER_TILE    = 1.0   # per new track tile visited
SPEED_REWARD_WEIGHT         = 0.000 # was 0.005 — slightly reduced so it prioritizes path over raw speed
MAX_SPEED_REWARD            = 0.0  # cap speed contribution so straights aren't over-rewarded
STEERING_SMOOTHNESS_WEIGHT  = 1.0  # was 0.15 — reduced so it's not afraid to turn the wheel sharp

REWARD_CLIP_RANGE       = None      # set e.g. (-1.0, 1.0) to clip; None = off

# ════════════════════════════════════════════════════════════════════════════════
# EVAL / LIVE RENDERING
# ════════════════════════════════════════════════════════════════════════════════

LIVE_RENDER         = False
LIVE_RENDER_EVERY   = 1
LIVE_RENDER_FPS     = 50

EVAL_RENDER         = True
EVAL_RENDER_FREQ    = 5_000
EVAL_RENDER_MAX_STEPS = 1_000
EVAL_RENDER_FPS     = 50

# ════════════════════════════════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


# ─── Environment builders ─────────────────────────────────────────────────────

def _env_kwargs():
    """Single source of truth for env construction shared by training and eval."""
    return dict(
        legacy_preprocessing        = False,
        grayscale_obs               = True,
        reward_shaping              = REWARD_SHAPING,
        # termination
        terminate_on_no_reward      = False,
        terminate_on_no_progress    = TERMINATE_NO_PROGRESS,
        max_no_progress_steps       = NO_PROGRESS_STEPS_LIMIT,
        termination_penalty         = TERMINATION_PENALTY,
        # per-step
        no_progress_penalty         = NO_PROGRESS_PENALTY,
        off_track_penalty           = OFF_TRACK_PENALTY,
        time_penalty                = TIME_PENALTY_PER_STEP,
        # bonuses
        completion_bonus            = COMPLETION_BONUS,
        # weights
        progress_reward_per_tile    = PROGRESS_REWARD_PER_TILE,
        speed_reward_weight         = SPEED_REWARD_WEIGHT,
        max_speed_reward            = MAX_SPEED_REWARD,
        steering_smoothness_weight  = STEERING_SMOOTHNESS_WEIGHT,
        # clipping
        reward_clip                 = REWARD_CLIP_RANGE,
    )


def make_env():
    def _init():
        return make_car_racing_env(render_mode=None, **_env_kwargs())
    return _init


def build_env():
    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns) if (USE_SUBPROC and NUM_ENVS > 1) else DummyVecEnv(env_fns)
    env = VecTransposeImage(env)        # (H, W, C) → (C, H, W) for CNN
    env = VecFrameStack(env, n_stack=4) # 4-frame stack so the agent sees motion
    return env


def build_eval_env():
    def _init():
        return make_car_racing_env(render_mode="rgb_array", **_env_kwargs())
    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


# ─── Render helpers ───────────────────────────────────────────────────────────

def _unwrap_render_env(env):
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def _get_training_render_env(training_env):
    raw = training_env
    for attr in ("venv", "venv"):
        if hasattr(raw, attr):
            raw = getattr(raw, attr)
    if hasattr(raw, "envs") and raw.envs:
        return _unwrap_render_env(raw.envs[0])
    return None


# ─── Callbacks ────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, print_freq=5_000, stats_window=200):
        super().__init__()
        self.total_timesteps  = total_timesteps
        self.print_freq       = print_freq
        self.last_print       = 0
        self.recent_rewards   = deque(maxlen=stats_window)
        self.recent_tiles     = deque(maxlen=stats_window)
        self.recent_speed     = deque(maxlen=stats_window)
        self.recent_completed = deque(maxlen=stats_window)

    def _on_step(self):
        reward = self.locals.get("rewards")
        if reward is not None:
            try:
                for r in reward:
                    self.recent_rewards.append(float(r))
            except (TypeError, ValueError):
                pass

        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                v = info.get("track_tiles_visited")
                if v is not None:
                    self.recent_tiles.append(int(v))
                s = info.get("car_speed")
                if s is not None:
                    self.recent_speed.append(float(s))
                c = info.get("track_completed")
                if c is not None:
                    self.recent_completed.append(1.0 if c else 0.0)

        if (
            self.num_timesteps - self.last_print >= self.print_freq
            or self.num_timesteps >= self.total_timesteps
        ):
            self.last_print = self.num_timesteps
            progress  = min(self.num_timesteps / self.total_timesteps, 1.0)
            bar_width = 30
            filled    = int(bar_width * progress)
            bar       = "#" * filled + "-" * (bar_width - filled)

            avg_reward     = sum(self.recent_rewards)   / len(self.recent_rewards)   if self.recent_rewards   else 0.0
            avg_tiles      = sum(self.recent_tiles)     / len(self.recent_tiles)     if self.recent_tiles     else 0.0
            avg_speed      = sum(self.recent_speed)     / len(self.recent_speed)     if self.recent_speed     else 0.0
            completion_rate = sum(self.recent_completed) / len(self.recent_completed) if self.recent_completed else 0.0

            self.logger.record("rollout/avg_reward",  avg_reward)
            self.logger.record("rollout/avg_tiles",   avg_tiles)
            self.logger.record("rollout/avg_speed",   avg_speed)
            self.logger.record("rollout/lap_rate",    completion_rate)
            print(
                f"\rProgress: [{bar}] "
                f"{self.num_timesteps:,} / {self.total_timesteps:,} steps "
                f"({progress * 100:5.1f}%) "
                f"avg_reward={avg_reward:6.3f} "
                f"avg_tiles={avg_tiles:6.1f} "
                f"avg_speed={avg_speed:5.2f} "
                f"lap_rate={completion_rate:4.2f}",
                end="", flush=True,
            )
        return True

    def _on_training_end(self):
        print()


class EvalRenderCallback(BaseCallback):
    def __init__(self, eval_freq, max_steps=1_000, fps=50):
        super().__init__()
        self.eval_freq  = eval_freq
        self.max_steps  = max_steps
        self.fps        = fps
        self.last_eval  = 0
        self.font       = None

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps

        eval_env = build_eval_env()
        obs      = eval_env.reset()
        raw_env  = _unwrap_render_env(eval_env.venv.envs[0])

        pygame.init()
        self.font = pygame.font.SysFont("Helvetica", 16)
        frame     = raw_env.render()
        if frame is None:
            eval_env.close()
            pygame.quit()
            return True
        h, w    = frame.shape[:2]
        screen  = pygame.display.set_mode((w, h))
        pygame.display.set_caption("PPO Eval")
        clock   = pygame.time.Clock()

        steps = 0; done = False; last_reward = 0.0; total_reward = 0.0
        while not done and steps < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True; break

            action, _ = self.model.predict(obs, deterministic=True)
            obs, step_reward, done, _info = eval_env.step(action)
            try:
                r_val = float(step_reward[0])
                last_reward   = r_val
                total_reward += r_val
            except (TypeError, ValueError, IndexError):
                pass

            frame = raw_env.render()
            if frame is None:
                break
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            if self.font is not None:
                label_step  = self.font.render(f"Reward (Step): {last_reward:+.3f}",  True, (255, 255, 0))
                label_total = self.font.render(f"Reward (Total): {total_reward:+.1f}", True, (255, 255, 0))
                screen.blit(label_step,  (8, frame.shape[0] - 88))
                screen.blit(label_total, (8, frame.shape[0] - 68))
            pygame.display.flip()
            clock.tick(self.fps)
            steps += 1

        eval_env.close()
        pygame.quit()
        return True


class LiveRenderCallback(BaseCallback):
    def __init__(self, every=1, fps=50):
        super().__init__()
        self.every        = max(1, int(every))
        self.fps          = fps
        self.clock        = None
        self.screen       = None
        self.enabled      = True
        self.font         = None
        self.total_rewards = None

    def _on_training_start(self):
        pygame.init()
        self.clock        = pygame.time.Clock()
        self.font         = pygame.font.SysFont("Helvetica", 16)
        self.total_rewards = [0.0] * self.training_env.num_envs

    def _on_step(self):
        if not self.enabled or (self.num_timesteps % self.every) != 0:
            return True

        rewards = self.locals.get("rewards")
        dones   = self.locals.get("dones")
        if rewards is not None:
            for i in range(len(self.total_rewards)):
                self.total_rewards[i] += float(rewards[i])
                if dones[i]:
                    self.total_rewards[i] = 0.0

        raw_env = _get_training_render_env(self.training_env)
        if raw_env is None:
            return True
        frame = raw_env.render()
        if frame is None:
            return True

        if self.screen is None:
            h, w        = frame.shape[:2]
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("PPO Live Training")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.enabled = False
                return True

        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))
        if rewards is not None and self.font is not None:
            try:
                step_val  = float(rewards[0])
                total_val = self.total_rewards[0]
                self.screen.blit(
                    self.font.render(f"Reward (Step): {step_val:+.3f}",  True, (255, 255, 0)),
                    (8, frame.shape[0] - 68),
                )
                self.screen.blit(
                    self.font.render(f"Reward (Total): {total_val:+.1f}", True, (255, 255, 0)),
                    (8, frame.shape[0] - 48),
                )
            except (TypeError, ValueError, IndexError):
                pass
        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.fps)
        return True

    def _on_training_end(self):
        pygame.quit()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    env = build_env()

    # Auto-resume from latest checkpoint if one exists
    #checkpoint_files = glob.glob(f"{SAVE_DIR}/{MODEL_NAME}_*_steps.zip")
    # Select specific checkpoint to resume from (e.g., n1_800k.zip) or use the latest available
    checkpoint_files = glob.glob(f"{SAVE_DIR}/n1_800k.zip")
    model = None
    if checkpoint_files:
        latest = max(checkpoint_files, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest}")
        candidate = PPO.load(latest, env=env, tensorboard_log=TENSORBOARD_LOG)
        if candidate.action_space.shape != env.action_space.shape:
            print("Checkpoint action space mismatch — starting fresh.")
        else:
            model = candidate

    if model is None:
        print("No usable checkpoint found — starting fresh.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
            learning_rate=3e-5,
            n_steps=1024,       # was 512 — increased to give better gradient estimates for complex turns
            batch_size=64,
            n_epochs=10,        # was 5 — increased to squeeze more out of each experience batch
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            target_kl=0.03,
        )

    callbacks = [
        CheckpointCallback(save_freq=SAVE_EVERY, save_path=SAVE_DIR, name_prefix=MODEL_NAME),
        ProgressCallback(TOTAL_TIMESTEPS),
    ]
    if LIVE_RENDER:
        callbacks.append(LiveRenderCallback(LIVE_RENDER_EVERY, LIVE_RENDER_FPS))
    if EVAL_RENDER:
        callbacks.append(EvalRenderCallback(EVAL_RENDER_FREQ, EVAL_RENDER_MAX_STEPS, EVAL_RENDER_FPS))
    callbacks = CallbackList(callbacks)

    print("Starting training...")
    print(f"  Steps this run : {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoints    : {SAVE_DIR}")
    print(f"  TensorBoard    : {TENSORBOARD_LOG or 'disabled (pip install tensorboard)'}")
    print("  Press Ctrl+C to stop early — latest checkpoint will be kept\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=False,
            tb_log_name="PPO_2",  # This ensures SB3 uses a unique suffix (PPO_1, PPO_2, etc.) for each session
        )
    except KeyboardInterrupt:
        print("\nTraining stopped early.")

    model.save(f"{SAVE_DIR}/{MODEL_NAME}_final")
    print(f"\nModel saved to {SAVE_DIR}/{MODEL_NAME}_final")


if __name__ == "__main__":
    main()
