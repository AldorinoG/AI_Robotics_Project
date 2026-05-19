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
        self.action_space = _convert_space(env.action_space)
        self.metadata = getattr(env, "metadata", {})
        self._pending_seed = None

    def reset(self, **kwargs):
        if "seed" not in kwargs and self._pending_seed is not None:
            kwargs["seed"] = self._pending_seed
            self._pending_seed = None
        obs, info = self.env.reset(**kwargs)
        return obs, info  # FIX 1: return both obs and info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        total_tiles = len(getattr(self.unwrapped, "track", []))
        visited_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        info["track_tiles_total"] = total_tiles
        info["track_tiles_visited"] = visited_tiles
        info["track_completed"] = total_tiles > 0 and visited_tiles >= total_tiles
        return obs, reward, terminated, truncated, info  # FIX 2: keep terminated and truncated separate

    def render(self):
        return self.env.render()

    def seed(self, seed=None):
        self._pending_seed = seed
        self.action_space.seed(seed)
        return [seed]


class CarRacingRewardWrapper(gymnasium.Wrapper):
    """Reward shaping to discourage spinning and reward real track progress."""

    def __init__(self, env):
        super().__init__(env)
        self.prev_tile_count = 0
        self.no_progress_steps = 0
        self.prev_steering = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_tile_count = getattr(self.unwrapped, "tile_visited_count", 0)
        self.no_progress_steps = 0
        self.prev_steering = 0.0
        return obs, info

    def step(self, action):
        before_tiles = getattr(self.unwrapped, "tile_visited_count", 0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        after_tiles = getattr(self.unwrapped, "tile_visited_count", before_tiles)
        new_tiles = max(after_tiles - before_tiles, 0)

        if new_tiles > 0:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if action is not None:
            raw_steering = float(action[0])
            steering = abs(raw_steering)
            gas = float(action[1])
            brake = float(action[2])
            car = getattr(self.unwrapped, "car", None)
            speed = 0.0
            if car is not None:
                velocity = car.hull.linearVelocity
                speed = float(np.sqrt(velocity.x**2 + velocity.y**2))

            steering_change = abs(raw_steering - self.prev_steering)
            steering_penalty = 0.04 * steering
            hard_steering_penalty = 0.12 * max(steering - 0.65, 0.0)
            steering_smoothness_penalty = 0.02 * steering_change
            low_speed_turn_penalty = 0.15 if steering > 0.65 and speed < 1.5 else 0.0
            spin_penalty = 0.20 if steering > 0.55 and self.no_progress_steps > 20 else 0.0
            stuck_penalty = 0.10 if self.no_progress_steps > 50 else 0.0
            brake_penalty = 0.05 * brake
            throttle_bonus = 0.04 * gas if brake < 0.1 else 0.0
            progress_bonus = 0.25 * new_tiles

            shaped_penalty = (
                steering_penalty
                + hard_steering_penalty
                + steering_smoothness_penalty
                + low_speed_turn_penalty
                + spin_penalty
                + stuck_penalty
                + brake_penalty
            )
            reward += progress_bonus + throttle_bonus - shaped_penalty
            info["reward_steering_penalty"] = shaped_penalty
            info["reward_progress_bonus"] = progress_bonus
            info["reward_throttle_bonus"] = throttle_bonus
            info["no_progress_steps"] = self.no_progress_steps
            info["car_speed"] = speed
            self.prev_steering = raw_steering

        return obs, reward, terminated, truncated, info


class CarRacingActionWrapper(gymnasium.ActionWrapper):
    """Limit extreme steering and avoid zero-throttle policies."""

    def __init__(self, env, max_steering=0.45, min_gas=0.18):
        super().__init__(env)
        self.max_steering = max_steering
        self.min_gas = min_gas

    def action(self, action):
        action = np.array(action, dtype=np.float32, copy=True)
        action[0] = np.clip(action[0], -self.max_steering, self.max_steering)

        # If the agent is not deliberately braking, make sure the car actually moves.
        if action[2] < 0.1:
            action[1] = max(action[1], self.min_gas)

        return action


class CarRacingLegacyWrapper(gymnasium.Wrapper):
    """Preprocessing/action mapping adapted from the copied working project."""

    def __init__(self, env, max_no_reward_steps=30):
        super().__init__(env)
        self.max_no_reward_steps = max_no_reward_steps
        self.t = 0
        self.last_reward_step = 0
        self.prev_steering = 0.0
        self.action_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, 1),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        self.t = 0
        self.last_reward_step = 0
        self.prev_steering = 0.0
        obs, info = self.env.reset(**kwargs)
        return self._postprocess_obs(obs), info

    def step(self, action):
        self.t += 1
        obs, reward, terminated, truncated, info = self.env.step(self._preprocess_action(action))

        if reward > 0:
            self.last_reward_step = self.t
        if (
            self.max_no_reward_steps is not None
            and self.t - self.last_reward_step > self.max_no_reward_steps
        ):
            terminated = True
            info["terminated_no_reward"] = True

        reward = float(np.clip(reward, -1.0, 1.0))
        return self._postprocess_obs(obs), reward, terminated, truncated, info

    def _preprocess_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        target_steering = action[0] * 2.0 - 1.0
        max_steering = self._max_steering_for_speed()
        target_steering = float(np.clip(target_steering, -max_steering, max_steering))
        steering = 0.75 * self.prev_steering + 0.25 * target_steering
        self.prev_steering = steering

        car_action = np.zeros(3, dtype=np.float32)
        turn_amount = abs(steering)
        gas = min(max(0.12, float(action[1])), 0.55)
        brake = 0.0
        if turn_amount > 0.22:
            gas = min(gas, 0.18)
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

        velocity = car.hull.linearVelocity
        speed = float(np.sqrt(velocity.x**2 + velocity.y**2))
        if speed < 2.0:
            return 0.25
        if speed < 5.0:
            return 0.30
        return 0.35

    def _postprocess_obs(self, obs):
        grayscale = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        obs = np.dot(obs[..., :3], grayscale).astype(np.uint8)
        return obs[..., None]


def make_car_racing_env(
    render_mode=None,
    max_episode_steps=None,
    reward_shaping=False,
    action_assist=False,
    legacy_preprocessing=False,
    terminate_on_no_reward=True,
):
    kwargs = {"continuous": True}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    env = gymnasium.make("CarRacing-v3", **kwargs)
    if legacy_preprocessing:
        max_no_reward_steps = 30 if terminate_on_no_reward else None
        env = CarRacingLegacyWrapper(env, max_no_reward_steps=max_no_reward_steps)
    elif action_assist:
        env = CarRacingActionWrapper(env)
    if reward_shaping:
        env = CarRacingRewardWrapper(env)
    return GymnasiumToGymWrapper(env)