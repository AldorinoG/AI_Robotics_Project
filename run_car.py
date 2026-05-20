import gymnasium as gym
import pygame
import tkinter as tk
from tkinter import font as tkfont
import time
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from env_compat import make_car_racing_env

SAVE_DIR             = "./models"
MODEL_NAME           = "racenet_cnn_ppo_legacy_final"
LEADERBOARD_FILE     = "leaderboard.json"
VIEWER_SIZE          = (900, 700)
MAX_RACE_STEPS       = 100000
TIME_LIMIT_SECONDS   = 20
LAP_COMPLETE_PERCENT  = 0.94  # must match CarRacing-v3 lap_complete_percent param
LAP_START_RADIUS      = 8.0   # world-unit distance to tile 0 that counts as crossing the line
LAP_MIN_PROGRESS      = 0.5   # must have visited at least 50% of tiles before line counts
MAP_OPTIONS = {
    "Map 1": 7,
    "Map 2": 21,
    "Map 3": 42,
    "Map 4": 84,
    "Map 5": 126,
}


class RaceViewer:
    def __init__(self, title):
        self.screen = pygame.display.set_mode(VIEWER_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.view_mode = "map"
        self.camera_zoom = 1.0
        self.camera_pan_x = 0
        self.camera_pan_y = 0
        self.map_zoom = None
        self.map_pan_x = 0
        self.map_pan_y = 0
        self.dragging = False
        self.last_mouse_pos = None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            self.view_mode = "camera" if self.view_mode == "map" else "map"
        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.15 if event.y > 0 else 1 / 1.15
            if self.view_mode == "map":
                if self.map_zoom is not None:
                    self.map_zoom = float(np.clip(self.map_zoom * factor, 0.5, 80.0))
            else:
                self.camera_zoom = float(np.clip(self.camera_zoom * factor, 0.4, 4.0))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = True
            self.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
            self.last_mouse_pos = None
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            if self.last_mouse_pos is not None:
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]
                if self.view_mode == "map":
                    self.map_pan_x += dx
                    self.map_pan_y += dy
                else:
                    self.camera_pan_x += dx
                    self.camera_pan_y += dy
            self.last_mouse_pos = event.pos
        return True

    def draw(self, frame, world_env=None, present=True):
        if self.view_mode == "map" and world_env is not None:
            self._draw_track_map(world_env, present)
        else:
            self._draw_camera(frame, present)

    def wait_for_home(self, frame, world_env, lines):
        clock = pygame.time.Clock()
        while True:
            self.draw(frame, world_env, present=False)
            home_rect = self._draw_message_overlay(lines)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if home_rect.collidepoint(event.pos):
                        return
                self.handle_event(event)
            clock.tick(30)

    def _draw_message_overlay(self, lines):
        screen_w, screen_h = self.screen.get_size()
        panel_w = min(520, screen_w - 40)
        panel_h = max(220, 150 + len(lines[1:]) * 30)
        panel_x = (screen_w - panel_w) // 2
        panel_y = (screen_h - panel_h) // 2
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, (13, 17, 23), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (15, 158, 117), panel_rect, 2, border_radius=8)

        title_font  = pygame.font.SysFont("Helvetica", 30, bold=True)
        text_font   = pygame.font.SysFont("Helvetica", 20)
        button_font = pygame.font.SysFont("Helvetica", 20, bold=True)

        title = title_font.render(lines[0], True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(screen_w // 2, panel_y + 42)))

        y = panel_y + 86
        for line in lines[1:]:
            text = text_font.render(line, True, (226, 232, 240))
            self.screen.blit(text, text.get_rect(center=(screen_w // 2, y)))
            y += 30

        home_rect = pygame.Rect(0, 0, 150, 46)
        home_rect.center = (screen_w // 2, panel_y + panel_h - 44)
        pygame.draw.rect(self.screen, (15, 158, 117), home_rect, border_radius=6)
        label = button_font.render("Home", True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=home_rect.center))

        pygame.display.flip()
        return home_rect

    def _draw_camera(self, frame, present=True):
        if frame is None:
            return
        frame_surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
        frame_w, frame_h = frame_surface.get_size()
        scaled_size = (
            max(1, int(frame_w * self.camera_zoom)),
            max(1, int(frame_h * self.camera_zoom)),
        )
        scaled_surface = pygame.transform.smoothscale(frame_surface, scaled_size)
        screen_w, screen_h = self.screen.get_size()
        x = (screen_w - scaled_size[0]) // 2 + self.camera_pan_x
        y = (screen_h - scaled_size[1]) // 2 + self.camera_pan_y
        self.screen.fill((13, 17, 23))
        self.screen.blit(scaled_surface, (x, y))
        if present:
            pygame.display.flip()

    def _draw_track_map(self, env, present=True):
        road_polygons = self._road_polygons(env)
        if not road_polygons:
            self._draw_camera(env.render(), present)
            return
        screen_w, screen_h = self.screen.get_size()
        if self.map_zoom is None:
            self._fit_map_to_screen(road_polygons, screen_w, screen_h)
        self.screen.fill((75, 170, 75))
        for polygon in road_polygons:
            points = [self._world_to_screen(x, y) for x, y in polygon]
            pygame.draw.polygon(self.screen, (95, 95, 95), points)
        track = getattr(env.unwrapped, "track", [])
        if len(track) > 1:
            centerline = [self._world_to_screen(x, y) for _alpha, _beta, x, y in track]
            pygame.draw.lines(self.screen, (180, 180, 180), True, centerline, 2)
            self._draw_track_coins(env, track)
        # Draw start/finish line marker at tile 0
        if len(track) > 0:
            _a, _b, sx, sy = track[0]
            fx, fy = self._world_to_screen(sx, sy)
            flag_r = max(6, int(self.map_zoom * 1.0))
            pygame.draw.circle(self.screen, (255, 255, 255), (fx, fy), flag_r + 2)
            pygame.draw.circle(self.screen, (30, 30, 30),   (fx, fy), flag_r)
            font = pygame.font.SysFont("Helvetica", max(10, flag_r * 2), bold=True)
            lbl  = font.render("S", True, (255, 255, 255))
            self.screen.blit(lbl, lbl.get_rect(center=(fx, fy)))

        car = getattr(env.unwrapped, "car", None)
        if car is not None:
            car_x, car_y = car.hull.position
            car_angle = -car.hull.angle
            cx, cy = self._world_to_screen(car_x, car_y)
            size = max(8, int(self.map_zoom * 1.4))
            car_points = [
                (cx + np.cos(car_angle) * size,       cy + np.sin(car_angle) * size),
                (cx + np.cos(car_angle + 2.4) * size, cy + np.sin(car_angle + 2.4) * size),
                (cx + np.cos(car_angle - 2.4) * size, cy + np.sin(car_angle - 2.4) * size),
            ]
            pygame.draw.polygon(self.screen, (220, 20, 20), car_points)
        if present:
            pygame.display.flip()

    def _fit_map_to_screen(self, road_polygons, screen_w, screen_h):
        xs = [x for polygon in road_polygons for x, _y in polygon]
        ys = [y for polygon in road_polygons for _x, y in polygon]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        map_w = max(max_x - min_x, 1)
        map_h = max(max_y - min_y, 1)
        self.map_zoom  = min(screen_w / map_w, screen_h / map_h) * 0.9
        self.map_pan_x = screen_w / 2 - ((min_x + max_x) / 2) * self.map_zoom
        self.map_pan_y = screen_h / 2 + ((min_y + max_y) / 2) * self.map_zoom

    def _world_to_screen(self, x, y):
        return (
            int(x  * self.map_zoom + self.map_pan_x),
            int(-y * self.map_zoom + self.map_pan_y),
        )

    def _road_polygons(self, env):
        polygons = []
        for road_piece in getattr(env.unwrapped, "road", []):
            for fixture in getattr(road_piece, "fixtures", []):
                vertices = getattr(fixture.shape, "vertices", None)
                if vertices:
                    polygons.append([(float(x), float(y)) for x, y in vertices])
        return polygons

    def _draw_track_coins(self, env, track):
        road_tiles  = getattr(env.unwrapped, "road", [])
        coin_radius = int(np.clip(self.map_zoom * 0.45, 3, 8))
        for index, (_alpha, _beta, x, y) in enumerate(track):
            road_tile = road_tiles[index] if index < len(road_tiles) else None
            collected = bool(getattr(road_tile, "road_visited", False))
            sx, sy = self._world_to_screen(x, y)
            color   = (22, 210, 90)  if collected else (245, 190, 35)
            outline = (4, 110, 48)   if collected else (140, 92, 8)
            pygame.draw.circle(self.screen, outline, (sx, sy), coin_radius + 1)
            pygame.draw.circle(self.screen, color,   (sx, sy), coin_radius)

# LEADERBOARD

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def leaderboard_score(entry):
    return int(entry.get("coins", 0))

def add_to_leaderboard(name, coins=None, lap_time=None, score_type="coins"):
    data = load_leaderboard()
    if score_type == "time" and lap_time is not None:
        best_by_name = {e["name"]: e for e in data if e.get("score_type") == "time"}
        existing = best_by_name.get(name)
        if existing is None or lap_time < existing.get("lap_time", float("inf")):
            best_by_name[name] = {"name": name, "score_type": "time", "lap_time": round(lap_time, 3), "coins": 0}
        coin_entries = [e for e in data if e.get("score_type", "coins") == "coins"]
        leaderboard  = coin_entries + list(best_by_name.values())
    else:
        coins       = int(coins) if coins is not None else 0
        coin_entries = [e for e in data if e.get("score_type", "coins") == "coins"]
        time_entries = [e for e in data if e.get("score_type") == "time"]
        best_by_name = {}
        for entry in coin_entries:
            entry_name = entry["name"]
            if entry_name not in best_by_name or leaderboard_score(entry) > leaderboard_score(best_by_name[entry_name]):
                best_by_name[entry_name] = {"name": entry_name, "score_type": "coins", "coins": leaderboard_score(entry)}
        if name not in best_by_name or coins > leaderboard_score(best_by_name[name]):
            best_by_name[name] = {"name": name, "score_type": "coins", "coins": coins}
        leaderboard = list(best_by_name.values()) + time_entries
    leaderboard.sort(key=leaderboard_score, reverse=True)
    save_leaderboard(leaderboard)
    return leaderboard


def _raw_unwrapped(env):
    """Walk the full wrapper chain to find the CarRacing env itself."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def track_progress_check(env):
    """Return (visited_tiles, total_tiles) by walking to the raw env."""
    raw     = _raw_unwrapped(env)
    total   = len(getattr(raw, "track", []))
    visited = getattr(raw, "tile_visited_count", 0)
    return visited, total


def get_start_pos(env):
    """Return (x, y) world coordinates of tile 0 (start/finish line)."""
    raw   = _raw_unwrapped(env)
    track = getattr(raw, "track", [])
    if not track:
        return None
    _alpha, _beta, x, y = track[0]
    return float(x), float(y)


def track_completed_check(env):
    """
    Lap complete when the car returns to within LAP_START_RADIUS of tile 0
    AND has already visited at least LAP_MIN_PROGRESS of the track.
    This mirrors how a real race works — cross the start/finish line to end.
    """
    raw     = _raw_unwrapped(env)
    track   = getattr(raw, "track", [])
    total   = len(track)
    visited = getattr(raw, "tile_visited_count", 0)

    if total == 0 or visited < int(total * LAP_MIN_PROGRESS):
        return False  # hasn't done enough of the lap yet

    car = getattr(raw, "car", None)
    if car is None:
        return False

    start_pos = get_start_pos(env)
    if start_pos is None:
        return False

    car_x, car_y   = car.hull.position
    start_x, start_y = start_pos
    dist = np.sqrt((car_x - start_x) ** 2 + (car_y - start_y) ** 2)
    return dist <= LAP_START_RADIUS

# AI DRIVE

def run_ai(map_seed, game_mode="coins"):
    print("Loading trained AI model...")

    import gymnasium as _gym
    import env_compat as _ec

    def make_gymnasium_env():
        def _init():
            env = _gym.make(
                "CarRacing-v3",
                render_mode="rgb_array",
                continuous=True,
                max_episode_steps=MAX_RACE_STEPS,
                lap_complete_percent=LAP_COMPLETE_PERCENT,
            )
            env = _ec.CarRacingLegacyWrapper(env, max_no_reward_steps=None)
            return env
        return _init

    from stable_baselines3.common.vec_env import (
        DummyVecEnv as _DVE, VecFrameStack as _VFS, VecTransposeImage as _VTI,
    )
    vec_env = _DVE([make_gymnasium_env()])
    vec_env = _VTI(vec_env)
    vec_env = _VFS(vec_env, n_stack=4)

    model = PPO.load(f"{SAVE_DIR}/{MODEL_NAME}", env=vec_env)

    vec_env.seed(map_seed)
    obs = vec_env.reset()

    # The inner env we check for lap state — walk past VecEnv layers
    inner_env = vec_env.venv.venv.envs[0]   # DummyVecEnv -> envs[0] = CarRacingLegacyWrapper

    start_time    = time.time()
    lap_time      = None
    coins         = 0
    total_tiles   = 0
    final_message = None
    timed_out     = False

    pygame.init()
    mode_label = "Collect Coins" if game_mode == "coins" else "Best Time"
    viewer = RaceViewer(f"Apex AI - AI Drive ({mode_label})")
    clock  = pygame.time.Clock()

    print("AI is driving.")
    print("Watch the simulation window.\n")

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if not viewer.handle_event(event):
                    running = False
            if not running:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # Render from the inner env (has render_mode set)
            viewer.draw(inner_env.env.render(), inner_env)
            clock.tick(50)

            elapsed             = time.time() - start_time
            coins, total_tiles  = track_progress_check(inner_env)
            lap_done_now        = track_completed_check(inner_env)

            print(
                f"\rAI time: {elapsed:.1f}s  tiles: {coins}/{total_tiles}"
                f"  done: {done[0]}  lap: {lap_done_now}",
                end="", flush=True,
            )

            if lap_done_now:
                lap_time = time.time() - start_time
                if game_mode == "time":
                    print(f"\nAI completed the lap in {lap_time:.2f}s")
                else:
                    print(f"\nLap complete. Score: {coins} coins")
                running = False

            elif done[0]:
                # Episode ended — was it a lap or a crash/timeout?
                if game_mode == "time":
                    final_message = [
                        "Driving Finished",
                        "Goal reached: No",
                        f"Track progress: {coins}/{total_tiles} tiles",
                        "Complete the full lap to set a time!",
                    ]
                else:
                    final_message = [
                        "Driving Finished",
                        "Goal reached: No",
                        f"Coins collected: {coins}/{total_tiles}",
                        "30s rule: more coins wins",
                    ]
                print(f"\nEpisode ended ({coins}/{total_tiles} tiles).")
                running = False

            elif game_mode == "coins" and elapsed >= TIME_LIMIT_SECONDS:
                timed_out = True
                final_message = [
                    "Driving Out Of Time",
                    f"Goal reached: No",
                    f"Coins collected: {coins}/{total_tiles}",
                    "30s rule: more coins wins",
                ]
                print(f"\nOut of time ({TIME_LIMIT_SECONDS}s limit).")
                running = False

            elif game_mode == "time" and elapsed >= 40:
                timed_out = True
                final_message = [
                    "Time's Up!",
                    "AI did not complete the lap in 40s.",
                    f"Track progress: {coins}/{total_tiles} tiles",
                ]
                print("\nAI ran out of time (40s limit).")
                running = False

    except KeyboardInterrupt:
        vec_env.close()
        pygame.quit()
        print("\nAI run stopped.")
        return

    if final_message is not None:
        viewer.wait_for_home(inner_env.env.render(), inner_env, final_message)

    vec_env.close()
    pygame.quit()

    leaderboard = load_leaderboard()

    if game_mode == "time":
        if lap_time is not None:
            print(f"\nAI finished the lap in {round(lap_time, 2)}s")
            humans     = [e for e in leaderboard if e["name"] != "AI" and e.get("score_type") == "time"]
            humans.sort(key=lambda e: e.get("lap_time", float("inf")))
            human_best = humans[0] if humans else None
            data       = add_to_leaderboard("AI", lap_time=lap_time, score_type="time")
        else:
            print("\nAI did not complete the lap.")
            human_best = None
            data       = leaderboard
        show_ai_result_time(lap_time, human_best, data)
    else:
        print(f"\nAI score: {coins} coins")
        humans     = [e for e in leaderboard if e["name"] != "AI" and e.get("score_type", "coins") == "coins"]
        humans.sort(key=leaderboard_score, reverse=True)
        human_best = humans[0] if humans else None
        data       = add_to_leaderboard("AI", coins=coins, score_type="coins")
        show_ai_result(coins, human_best, data)


def show_ai_result(ai_coins, human_best, all_scores):
    root = tk.Tk()
    root.title("AI Race Result")
    root.geometry("500x420")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    big_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")

    tk.Label(root, text="AI Race Result", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 16))
    tk.Label(root, text=f"AI Coins:   {ai_coins}", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)

    if human_best:
        human_coins = leaderboard_score(human_best)
        tk.Label(root, text=f"Human Best:   {human_coins} coins  ({human_best['name']})", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)
        if ai_coins > human_coins:
            verdict, color = f"AI WINS by {ai_coins - human_coins} coins", "#0F9E75"
        elif ai_coins == human_coins:
            verdict, color = "Tie score", "#D97706"
        else:
            verdict, color = f"Human still wins by {human_coins - ai_coins} coins", "#D97706"
        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(20, 4))
    else:
        tk.Label(root, text="No human scores recorded yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text="Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_scores[:5]):
        rank  = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()


def show_ai_result_time(ai_lap_time, human_best, all_scores):
    root = tk.Tk()
    root.title("AI Race Result — Best Time")
    root.geometry("500x420")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    big_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")

    tk.Label(root, text="AI Race Result", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 16))

    if ai_lap_time is not None:
        tk.Label(root, text=f"AI Lap Time:   {ai_lap_time:.2f}s", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)
    else:
        tk.Label(root, text="AI did not complete the lap.", font=big_font, bg="#0D1117", fg="#DC2626").pack(pady=4)

    if human_best and ai_lap_time is not None:
        human_time = human_best.get("lap_time", None)
        tk.Label(root, text=f"Human Best:   {human_time:.2f}s  ({human_best['name']})", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)
        if ai_lap_time < human_time:
            verdict, color = f"AI WINS by {human_time - ai_lap_time:.2f}s", "#0F9E75"
        elif ai_lap_time == human_time:
            verdict, color = "Tie!", "#D97706"
        else:
            verdict, color = f"Human still wins by {ai_lap_time - human_time:.2f}s", "#D97706"
        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(20, 4))
    else:
        tk.Label(root, text="No human times recorded yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text="Best Times Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
    time_entries = [e for e in all_scores if e.get("score_type") == "time"]
    time_entries.sort(key=lambda e: e.get("lap_time", float("inf")))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(time_entries[:5]):
        rank  = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {entry.get('lap_time', 0):.2f}s", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()

# GUI LAUNCHER

def launch_gui():
    first_map_name = next(iter(MAP_OPTIONS))
    result = {"name": None, "mode": None, "map_seed": MAP_OPTIONS[first_map_name], "game_mode": "coins"}

    root = tk.Tk()
    root.title("Apex AI — Launcher")
    root.geometry("500x560")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=26, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=13, weight="bold")
    label_font = tkfont.Font(family="Helvetica", size=11)

    tk.Label(root, text="Apex AI",                       font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(30, 4))
    tk.Label(root, text="AI Racing — Human vs Machine",  font=sub_font,   bg="#0D1117", fg="#64748B").pack(pady=(0, 24))
    tk.Label(root, text="Enter your name:",              font=label_font, bg="#0D1117", fg="#FFFFFF").pack()

    name_var   = tk.StringVar()
    name_entry = tk.Entry(root, textvariable=name_var, font=label_font, width=28,
                          bg="#1E293B", fg="#FFFFFF", insertbackground="#FFFFFF", relief="flat", bd=8)
    name_entry.pack(pady=(6, 24))
    name_entry.focus()

    tk.Label(root, text="Choose map:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()
    map_var  = tk.StringVar(value=first_map_name)
    map_menu = tk.OptionMenu(root, map_var, *MAP_OPTIONS.keys())
    map_menu.config(font=label_font, width=24, bg="#1E293B", fg="#FFFFFF",
                    activebackground="#334155", activeforeground="#FFFFFF", relief="flat", highlightthickness=0)
    map_menu["menu"].config(bg="#1E293B", fg="#FFFFFF", activebackground="#334155")
    map_menu.pack(pady=(6, 18))

    tk.Label(root, text="Game mode:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()
    GAME_MODES     = {"Collect Coins": "coins", "Best Time": "time"}
    game_mode_var  = tk.StringVar(value="Collect Coins")
    game_mode_menu = tk.OptionMenu(root, game_mode_var, *GAME_MODES.keys())
    game_mode_menu.config(font=label_font, width=24, bg="#1E293B", fg="#FFFFFF",
                          activebackground="#334155", activeforeground="#FFFFFF", relief="flat", highlightthickness=0)
    game_mode_menu["menu"].config(bg="#1E293B", fg="#FFFFFF", activebackground="#334155")
    game_mode_menu.pack(pady=(6, 18))

    error_label = tk.Label(root, text="", font=sub_font, bg="#0D1117", fg="#DC2626")
    error_label.pack()

    def start_human():
        name = name_var.get().strip()
        if not name:
            error_label.config(text="Please enter your name before starting.")
            return
        result["name"]      = name
        result["mode"]      = "human"
        result["map_seed"]  = MAP_OPTIONS[map_var.get()]
        result["game_mode"] = GAME_MODES[game_mode_var.get()]
        root.destroy()

    tk.Button(root, text="Drive Manually", font=btn_font, command=start_human,
              bg="#0F9E75", fg="#FFFFFF", activebackground="#0A6E52", activeforeground="#FFFFFF",
              relief="flat", padx=20, pady=10, width=22).pack(pady=(8, 8))

    def start_ai():
        if not os.path.exists(f"{SAVE_DIR}/{MODEL_NAME}.zip"):
            error_label.config(text="No trained model found. Run train.py first.")
            return
        result["mode"]      = "ai"
        result["map_seed"]  = MAP_OPTIONS[map_var.get()]
        result["game_mode"] = GAME_MODES[game_mode_var.get()]
        root.destroy()

    tk.Button(root, text="Let AI Drive", font=btn_font, command=start_ai,
              bg="#1E293B", fg="#64748B", activebackground="#1E293B", activeforeground="#64748B",
              relief="flat", padx=20, pady=10, width=22).pack(pady=(0, 8))

    data = load_leaderboard()
    if data:
        coin_scores = sorted([e for e in data if e.get("score_type", "coins") == "coins"], key=leaderboard_score, reverse=True)
        time_scores = sorted([e for e in data if e.get("score_type") == "time"], key=lambda e: e.get("lap_time", float("inf")))
        if coin_scores:
            tk.Label(root, text="Top Coin Scores", font=label_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
            for i, entry in enumerate(coin_scores[:3]):
                tk.Label(root, text=f"{['1st','2nd','3rd'][i]}  {entry['name']}  —  {leaderboard_score(entry)} coins",
                         font=sub_font, bg="#0D1117", fg="#FFFFFF").pack()
        if time_scores:
            tk.Label(root, text="Top Best Times", font=label_font, bg="#0D1117", fg="#D97706").pack(pady=(8, 4))
            for i, entry in enumerate(time_scores[:3]):
                tk.Label(root, text=f"{['1st','2nd','3rd'][i]}  {entry['name']}  —  {entry.get('lap_time','?'):.2f}s",
                         font=sub_font, bg="#0D1117", fg="#FFFFFF").pack()

    root.mainloop()
    return result

# LEADERBOARD SCREEN

def show_leaderboard_screen(name, coins, all_scores):
    root = tk.Tk()
    root.title("Race Finished")
    root.geometry("500x460")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=12, weight="bold")
    row_font   = tkfont.Font(family="Helvetica", size=12)

    tk.Label(root, text="Race Finished",              font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 4))
    tk.Label(root, text=f"{name}  —  {coins} coins", font=sub_font,   bg="#0D1117", fg="#FFFFFF").pack(pady=(0, 20))
    tk.Label(root, text="Leaderboard",               font=sub_font,   bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_scores[:8]):
        rank   = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and leaderboard_score(entry) == coins
        color  = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins{suffix}",
                 font=row_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=btn_font, command=root.destroy,
              bg="#1E293B", fg="#FFFFFF", activebackground="#334155", relief="flat", padx=16, pady=8).pack(pady=(20, 0))
    root.mainloop()

# BEST TIME RESULT SCREEN

def show_time_result_screen(name, lap_time, all_scores):
    root = tk.Tk()
    root.title("Best Time Result")
    root.geometry("500x460")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=12, weight="bold")
    row_font   = tkfont.Font(family="Helvetica", size=12)

    tk.Label(root, text="Lap Complete!",              font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 4))
    tk.Label(root, text=f"{name}  —  {lap_time:.2f}s", font=sub_font, bg="#0D1117", fg="#FFFFFF").pack(pady=(0, 20))
    tk.Label(root, text="Best Times Leaderboard",    font=sub_font,   bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    time_entries = sorted([e for e in all_scores if e.get("score_type") == "time"],
                          key=lambda e: e.get("lap_time", float("inf")))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(time_entries[:8]):
        rank   = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and abs(entry.get("lap_time", -1) - lap_time) < 0.01
        color  = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {entry.get('lap_time', 0):.2f}s{suffix}",
                 font=row_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=btn_font, command=root.destroy,
              bg="#1E293B", fg="#FFFFFF", activebackground="#334155", relief="flat", padx=16, pady=8).pack(pady=(20, 0))
    root.mainloop()

# MAIN GAME (human)

def run_game(player_name, map_seed, game_mode="coins"):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        max_episode_steps=MAX_RACE_STEPS,
        lap_complete_percent=LAP_COMPLETE_PERCENT,
    )
    obs, info = env.reset(seed=map_seed)

    pygame.init()
    mode_label = "Collect Coins" if game_mode == "coins" else "Best Time"
    viewer = RaceViewer(f"Apex AI - Manual Drive ({mode_label})")
    clock  = pygame.time.Clock()

    print(f"\nWelcome {player_name}. Mode: {mode_label}")
    print("Controls: W = gas | S = brake | A = left | D = right | Q = quit")
    if game_mode == "coins":
        print(f"Timer starts on first W press. Collect coins in {TIME_LIMIT_SECONDS}s!\n")
    else:
        print("Timer starts on first W press. Complete the full lap as fast as you can!\n")

    start_time    = None
    lap_done      = False
    lap_time      = None
    coins         = 0
    timed_out     = False
    final_message = None

    running = True
    while running:
        action = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        for event in pygame.event.get():
            if not viewer.handle_event(event):
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: action[0] = -1.0
        if keys[pygame.K_d]: action[0] =  1.0
        if keys[pygame.K_w]:
            action[1] = 1.0
            if start_time is None:
                start_time = time.time()
                print("Timer started.")
        if keys[pygame.K_s]: action[2] = 0.8
        if keys[pygame.K_q]: running = False

        obs, reward, terminated, truncated, info = env.step(action)
        viewer.draw(env.render(), env)
        clock.tick(50)

        if start_time is not None and not lap_done:
            elapsed        = time.time() - start_time
            coins, total_t = track_progress_check(env)
            print(f"\r{elapsed:.1f}s  tiles: {coins}/{total_t}", end="", flush=True)

            if track_completed_check(env) or (terminated and not info.get("TimeLimit.truncated", False)):
                lap_time = time.time() - start_time
                lap_done = True
                print(f"\nLap complete! Time: {lap_time:.2f}s  Tiles: {coins}/{total_t}")
                running = False

            elif game_mode == "coins" and elapsed >= TIME_LIMIT_SECONDS:
                timed_out     = True
                final_message = [
                    "Driving Out Of Time", "Goal reached: No",
                    f"Coins collected: {coins}/{total_t}", "30s rule: more coins wins",
                ]
                print(f"\nOut of time ({TIME_LIMIT_SECONDS}s).")
                running = False

            elif game_mode == "time" and elapsed >= 40:
                timed_out     = True
                final_message = [
                    "Time's Up!", "You did not complete the lap in 40s.",
                    f"Track progress: {coins}/{total_t} tiles",
                ]
                print("\nRan out of time (40s).")
                running = False

            elif (terminated or truncated) and not lap_done:
                visited, total_t = track_progress_check(env)
                coins = visited
                final_message = [
                    "Driving Finished", "Goal reached: No",
                    f"Coins collected: {visited}/{total_t}" if game_mode == "coins"
                    else f"Track progress: {visited}/{total_t} tiles",
                    "30s rule: more coins wins" if game_mode == "coins"
                    else "Complete the full lap to set a time!",
                ]
                print(f"\nEpisode ended ({visited}/{total_t} tiles).")
                running = False

    if final_message is not None:
        viewer.wait_for_home(env.render(), env, final_message)

    env.close()
    pygame.quit()
    if game_mode == "time":
        return ("time", lap_time) if lap_done else (None, None)
    return ("coins", coins) if start_time is not None else (None, None)

# ENTRY POINT

if __name__ == "__main__":
    while True:
        result = launch_gui()

        if result["mode"] is None:
            print("No mode selected. Exiting.")
            break

        if result["mode"] == "human":
            player_name = result["name"]
            game_mode   = result.get("game_mode", "coins")
            score_type, score_value = run_game(player_name, result["map_seed"], game_mode)

            if score_type == "coins" and score_value is not None:
                all_scores = add_to_leaderboard(player_name, coins=score_value, score_type="coins")
                show_leaderboard_screen(player_name, score_value, all_scores)
            elif score_type == "time" and score_value is not None:
                all_scores = add_to_leaderboard(player_name, lap_time=score_value, score_type="time")
                show_time_result_screen(player_name, score_value, all_scores)
            else:
                print("No score recorded — returning to menu.")

        elif result["mode"] == "ai":
            run_ai(result["map_seed"], result.get("game_mode", "coins"))