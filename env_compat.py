import gym
import gymnasium as gymnasium
import numpy as np


def _convert_space(space):
    """Convert Gymnasium spaces to equivalent Gym spaces for older SB3 builds."""
    if isinstance(space, gymnasium.spaces.Box):
        return gym.spaces.Box(
            low=np.asarray(space.low),
            high=np.asarray(space.high),
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(space.n)
    if isinstance(space, gymnasium.spaces.MultiBinary):
        return gym.spaces.MultiBinary(space.n)
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(space.nvec)
    if isinstance(space, gymnasium.spaces.Dict):
        return gym.spaces.Dict({key: _convert_space(value) for key, value in space.spaces.items()})
    if isinstance(space, gymnasium.spaces.Tuple):
        return gym.spaces.Tuple(tuple(_convert_space(value) for value in space.spaces))
    raise TypeError(f"Unsupported Gymnasium space: {space!r}")


class GymnasiumToGymWrapper(gym.Wrapper):
    """Adapt a Gymnasium env to the Gym API used by this Stable-Baselines3 install."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = _convert_space(env.observation_space)
        self.action_space      = _convert_space(env.action_space)
        self.metadata          = getattr(env, "metadata", {})
        self._pending_seed     = None

    def reset(self, **kwargs):
        if "seed" not in kwargs and self._pending_seed is not None:
            kwargs["seed"] = self._pending_seed
            self._pending_seed = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        total_tiles   = len(getattr(self.unwrapped, "track", []))
        visited_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        info["track_tiles_total"]   = total_tiles
        info["track_tiles_visited"] = visited_tiles
        info["track_completed"]     = total_tiles > 0 and visited_tiles >= total_tiles
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def seed(self, seed=None):
        self._pending_seed = seed
        self.action_space.seed(seed)
        return [seed]


# ─── Reward Shaping ───────────────────────────────────────────────────────────

class CarRacingRewardWrapper(gymnasium.Wrapper):
    """
    Reward shaping: tile progress, speed, steering smoothness, time, and
    hard termination when the car stops making progress (fixes donut-circling).

    All tuning constants come from train.py via make_car_racing_env() —
    nothing is hardcoded here.
    """

    def __init__(
        self,
        env,
        # Progress / termination
        terminate_on_no_progress=True,
        max_no_progress_steps=25,
        termination_penalty=5.0,
        no_progress_penalty=0.3,
        off_track_penalty=0.0,
        # Episode bonuses / penalties
        time_penalty=0.0,
        completion_bonus=0.0,
        # Reward weights
        progress_reward_per_tile=1.0,
        speed_reward_weight=0.005,
        max_speed_reward=30.0,
        steering_smoothness_weight=0.15,
    ):
        super().__init__(env)
        # termination
        self.terminate_on_no_progress  = bool(terminate_on_no_progress)
        self.max_no_progress_steps      = int(max_no_progress_steps)
        self.termination_penalty        = float(termination_penalty)
        self.no_progress_penalty        = float(no_progress_penalty)
        self.off_track_penalty          = float(off_track_penalty)
        # episode-level
        self.time_penalty               = float(time_penalty)
        self.completion_bonus           = float(completion_bonus)
        # weights
        self.progress_reward_per_tile   = float(progress_reward_per_tile)
        self.speed_reward_weight        = float(speed_reward_weight)
        self.max_speed_reward           = float(max_speed_reward)
        self.steering_smoothness_weight = float(steering_smoothness_weight)
        # state
        self.no_progress_steps = 0
        self.prev_steering     = 0.0
        # CarRacing-v3 starts with tile_visited_count=2 during the zoom-in
        # warmup (≈50–80 steps where the car has no control).  We must not
        # start counting no-progress steps until the car has moved past that
        # initial value, otherwise every episode terminates in <1 second.
        self._initial_tiles   = None   # set on first step after reset
        self._warmup_done     = False  # True once tile count exceeds initial

    def reset(self, **kwargs):
        obs, info              = self.env.reset(**kwargs)
        self.no_progress_steps = 0
        self.prev_steering     = 0.0
        self._initial_tiles    = getattr(self.unwrapped, "tile_visited_count", 0)
        self._warmup_done      = False
        return obs, info

    def step(self, action):
        before_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        after_tiles  = getattr(self.unwrapped, "tile_visited_count", before_tiles)
        new_tiles    = max(after_tiles - before_tiles, 0)

        # ── Warmup guard ──────────────────────────────────────────────────────
        # Don't count no-progress steps until the car has driven past the
        # initial tile count that CarRacing-v3 pre-seeds at episode start.
        if not self._warmup_done:
            if after_tiles > self._initial_tiles:
                self._warmup_done = True
            # During warmup: reset counter and skip all no-progress logic
            self.no_progress_steps = 0

        # ── Progress counter (only active after warmup) ───────────────────────
        if self._warmup_done:
            if new_tiles > 0:
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1

        # ── Per-step shaped rewards ───────────────────────────────────────────
        shaped = 0.0

        if action is not None:
            raw_steering  = float(action[0])
            steering_delta = abs(raw_steering - self.prev_steering)

            # Speed (encourages actually moving, not spinning in place)
            car   = getattr(self.unwrapped, "car", None)
            speed = 0.0
            if car is not None:
                v     = car.hull.linearVelocity
                speed = float(np.sqrt(v.x ** 2 + v.y ** 2))

            progress_bonus            = self.progress_reward_per_tile * new_tiles
            speed_reward              = self.speed_reward_weight * min(speed, self.max_speed_reward)
            steering_smoothness_penalty = self.steering_smoothness_weight * steering_delta

            shaped += progress_bonus + speed_reward - steering_smoothness_penalty

            info["reward_progress_bonus"]          = progress_bonus
            info["reward_speed_bonus"]             = speed_reward
            info["reward_steering_penalty"]        = steering_smoothness_penalty
            info["no_progress_steps"]              = self.no_progress_steps
            info["car_speed"]                      = speed
            self.prev_steering = raw_steering

        # ── Off-track penalty ─────────────────────────────────────────────────
        if self.off_track_penalty:
            # Check if all 4 wheels are off the track
            car = getattr(self.unwrapped, "car", None)
            if car is not None:
                on_track = False
                for wheel in car.wheels:
                    if len(wheel.tiles) > 0:
                        on_track = True
                        break
                if not on_track:
                    shaped -= self.off_track_penalty
                    info["reward_off_track_penalty"] = self.off_track_penalty

        # ── No-progress per-step penalty (only after warmup) ─────────────────
        if self._warmup_done and new_tiles == 0 and self.no_progress_penalty:
            shaped -= self.no_progress_penalty
            info["reward_no_progress_penalty"] = self.no_progress_penalty

        # ── Time penalty ──────────────────────────────────────────────────────
        if self.time_penalty:
            shaped -= self.time_penalty
            info["reward_time_penalty"] = self.time_penalty

        # ── Completion bonus ──────────────────────────────────────────────────
        total_tiles   = len(getattr(self.unwrapped, "track", []))
        visited_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        if self.completion_bonus and total_tiles > 0 and visited_tiles >= total_tiles:
            shaped += self.completion_bonus
            info["reward_completion_bonus"] = self.completion_bonus

        # ── Hard termination for no-progress (only after warmup) ────────────
        if self._warmup_done and self.terminate_on_no_progress and self.no_progress_steps >= self.max_no_progress_steps:
            terminated = True
            info["terminated_no_progress"] = True
            if self.termination_penalty:
                shaped -= self.termination_penalty
                info["reward_termination_penalty"] = self.termination_penalty

        reward += shaped
        return obs, reward, terminated, truncated, info


# ─── Action Wrappers ──────────────────────────────────────────────────────────

class CarRacingActionWrapper(gymnasium.ActionWrapper):
    """Limit extreme steering and prevent zero-throttle policies."""

    def __init__(self, env, max_steering=0.45, min_gas=0.18):
        super().__init__(env)
        self.max_steering = max_steering
        self.min_gas      = min_gas

    def action(self, action):
        action    = np.array(action, dtype=np.float32, copy=True)
        action[0] = np.clip(action[0], -self.max_steering, self.max_steering)
        if action[2] < 0.1:
            action[1] = max(action[1], self.min_gas)
        return action


# ─── Observation Wrappers ─────────────────────────────────────────────────────

class CarRacingGrayScaleWrapper(gymnasium.ObservationWrapper):
    """Convert RGB observations to grayscale (H, W, 1)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(96, 96, 1), dtype=np.uint8,
        )

    def observation(self, obs):
        gray = np.dot(obs[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        return gray.astype(np.uint8)[..., None]


# ─── Legacy preprocessing (kept for backward-compat) ─────────────────────────

class CarRacingLegacyWrapper(gymnasium.Wrapper):
    """Preprocessing / action mapping from the original working project."""

    def __init__(
        self,
        env,
        max_no_reward_steps=30,
        max_no_progress_steps=None,
        reward_clip=(-1.0, 1.0),
        termination_penalty=0.0,
    ):
        super().__init__(env)
        self.max_no_reward_steps   = max_no_reward_steps
        self.max_no_progress_steps = max_no_progress_steps
        self.reward_clip           = reward_clip
        self.termination_penalty   = float(termination_penalty)
        self.t                     = 0
        self.last_reward_step      = 0
        self.last_progress_step    = 0
        self.prev_steering         = 0.0
        self.action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(96, 96, 1), dtype=np.uint8,
        )

    def reset(self, **kwargs):
        self.t                  = 0
        self.last_reward_step   = 0
        self.last_progress_step = 0
        self.prev_steering      = 0.0
        obs, info = self.env.reset(**kwargs)
        return self._postprocess_obs(obs), info

    def step(self, action):
        self.t += 1
        before_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        obs, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action))
        after_tiles  = getattr(self.unwrapped, "tile_visited_count", before_tiles)

        if reward > 0:
            self.last_reward_step = self.t
        if (
            self.max_no_reward_steps is not None
            and self.t - self.last_reward_step > self.max_no_reward_steps
        ):
            terminated = True
            info["terminated_no_reward"] = True

        if after_tiles > before_tiles:
            self.last_progress_step = self.t
        if (
            self.max_no_progress_steps is not None
            and self.t - self.last_progress_step > self.max_no_progress_steps
        ):
            terminated = True
            info["terminated_no_progress"] = True
            if self.termination_penalty:
                reward -= self.termination_penalty
                info["reward_termination_penalty"] = self.termination_penalty

        if self.reward_clip is not None:
            reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))
        return self._postprocess_obs(obs), reward, terminated, truncated, info

    def _preprocess_action(self, action):
        action         = np.asarray(action, dtype=np.float32)
        target_steering = action[0] * 2.0 - 1.0
        max_steering    = self._max_steering_for_speed()
        target_steering = float(np.clip(target_steering, -max_steering, max_steering))
        steering        = 0.75 * self.prev_steering + 0.25 * target_steering
        self.prev_steering = steering

        car_action  = np.zeros(3, dtype=np.float32)
        turn_amount = abs(steering)
        gas         = min(max(0.12, float(action[1])), 0.55)
        brake       = 0.0
        if turn_amount > 0.22:
            gas   = min(gas, 0.18)
            brake = 0.15
        elif turn_amount > 0.08:
            gas = min(gas, 0.25)

        car_action[0] = steering
        car_action[1] = gas
        car_action[2] = brake
        return car_action

    def _max_steering_for_speed(self):
        car = getattr(self.unwrapped, "car", None)
        if car is None:
            return 0.35
        v     = car.hull.linearVelocity
        speed = float(np.sqrt(v.x ** 2 + v.y ** 2))
        if speed < 2.0:
            return 0.25
        if speed < 5.0:
            return 0.30
        return 0.35

    def _postprocess_obs(self, obs):
        gray = np.dot(obs[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        return gray.astype(np.uint8)[..., None]


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_car_racing_env(
    render_mode=None,
    max_episode_steps=None,
    lap_complete_percent=None,
    # Preprocessing mode
    legacy_preprocessing=False,
    grayscale_obs=False,
    action_assist=False,
    # Termination
    terminate_on_no_reward=True,
    terminate_on_no_progress=False,
    max_no_progress_steps=25,
    termination_penalty=5.0,
    # Per-step penalties
    no_progress_penalty=0.3,
    off_track_penalty=0.0,
    time_penalty=0.0,
    # Bonuses
    reward_shaping=False,
    completion_bonus=0.0,
    # Reward weights (passed through to CarRacingRewardWrapper)
    progress_reward_per_tile=1.0,
    speed_reward_weight=0.005,
    max_speed_reward=30.0,
    steering_smoothness_weight=0.15,
    # Legacy-only
    reward_clip=(-1.0, 1.0),
):
    kwargs = {"continuous": True}
    if render_mode       is not None: kwargs["render_mode"]        = render_mode
    if max_episode_steps is not None: kwargs["max_episode_steps"]  = max_episode_steps
    if lap_complete_percent is not None: kwargs["lap_complete_percent"] = lap_complete_percent

    env = gymnasium.make("CarRacing-v3", **kwargs)

    if legacy_preprocessing:
        max_no_reward_steps = 30 if terminate_on_no_reward else None
        if terminate_on_no_progress and max_no_progress_steps is None:
            max_no_progress_steps = 40
        env = CarRacingLegacyWrapper(
            env,
            max_no_reward_steps=max_no_reward_steps,
            max_no_progress_steps=max_no_progress_steps,
            reward_clip=reward_clip,
            termination_penalty=termination_penalty,
        )
    else:
        if action_assist:
            env = CarRacingActionWrapper(env)
        if grayscale_obs:
            env = CarRacingGrayScaleWrapper(env)

    if reward_shaping and not legacy_preprocessing:
        env = CarRacingRewardWrapper(
            env,
            terminate_on_no_progress=terminate_on_no_progress,
            max_no_progress_steps=max_no_progress_steps,
            termination_penalty=termination_penalty,
            no_progress_penalty=no_progress_penalty,
            off_track_penalty=off_track_penalty,
            time_penalty=time_penalty,
            completion_bonus=completion_bonus,
            progress_reward_per_tile=progress_reward_per_tile,
            speed_reward_weight=speed_reward_weight,
            max_speed_reward=max_speed_reward,
            steering_smoothness_weight=steering_smoothness_weight,
        )

    return GymnasiumToGymWrapper(env)
