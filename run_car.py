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
TIME_LIMIT_SECONDS   = 30
LAP_COMPLETE_PERCENT = 0.94
LAP_START_RADIUS     = 8.0
LAP_MIN_PROGRESS     = 0.5


MAP_OPTIONS = {
    "Map 1": 21,
    "Map 2": 84,
    "Map 3": 126,
}


# ─── RaceViewer ───────────────────────────────────────────────────────────────

MINIMAP_SIZE   = 200   # pixel size of the minimap square
MINIMAP_MARGIN = 12    # gap from screen edge
MINIMAP_ALPHA  = 200   # transparency (0=invisible, 255=opaque)

class RaceViewer:
    def __init__(self, title):
        self.screen       = pygame.display.set_mode(VIEWER_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.camera_zoom  = 1.5  # scale up the 96x96 render to fill the window
        self.camera_pan_x = 0
        self.camera_pan_y = 0
        # Minimap world-bounds (computed once when track is first seen)
        self._mini_world_bounds = None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.MOUSEWHEEL:
            factor = 1.15 if event.y > 0 else 1 / 1.15
            self.camera_zoom = float(np.clip(self.camera_zoom * factor, 0.4, 4.0))
        return True

    def draw(self, frame, world_env=None, present=True):
        self._draw_camera(frame)
        if world_env is not None:
            self._draw_minimap(world_env)
        if present:
            pygame.display.flip()

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
        pygame.draw.rect(self.screen, (13, 17, 23),   panel_rect, border_radius=8)
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

    def _draw_camera(self, frame):
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

    def _draw_minimap(self, env):
        """Draw a small map overlay in the bottom-right corner."""
        road_polygons = self._road_polygons(env)
        if not road_polygons:
            return

        # Compute world bounds once
        if self._mini_world_bounds is None:
            xs = [x for poly in road_polygons for x, _y in poly]
            ys = [y for poly in road_polygons for _x, y in poly]
            self._mini_world_bounds = (min(xs), max(xs), min(ys), max(ys))

        min_x, max_x, min_y, max_y = self._mini_world_bounds
        world_w = max(max_x - min_x, 1)
        world_h = max(max_y - min_y, 1)
        scale   = (MINIMAP_SIZE - 8) / max(world_w, world_h)

        # Offscreen surface for the minimap contents
        mini = pygame.Surface((MINIMAP_SIZE, MINIMAP_SIZE), pygame.SRCALPHA)
        mini.fill((20, 20, 28, MINIMAP_ALPHA))

        def to_mini(wx, wy):
            px = int((wx - min_x) * scale) + 4
            py = int((max_y - wy) * scale) + 4
            return px, py

        # Road
        for poly in road_polygons:
            pts = [to_mini(wx, wy) for wx, wy in poly]
            if len(pts) >= 3:
                pygame.draw.polygon(mini, (80, 80, 80), pts)

        # Track centerline + coins
        track      = getattr(env.unwrapped, "track", [])
        road_tiles = getattr(env.unwrapped, "road", [])
        if len(track) > 1:
            centerline = [to_mini(wx, wy) for _a, _b, wx, wy in track]
            pygame.draw.lines(mini, (140, 140, 140), True, centerline, 1)

        for idx, (_a, _b, wx, wy) in enumerate(track):
            road_tile = road_tiles[idx] if idx < len(road_tiles) else None
            collected = bool(getattr(road_tile, "road_visited", False))
            color = (22, 210, 90) if collected else (245, 190, 35)
            px, py = to_mini(wx, wy)
            pygame.draw.circle(mini, color, (px, py), 2)

        # Start/finish marker
        if track:
            _a, _b, sx, sy = track[0]
            fx, fy = to_mini(sx, sy)
            pygame.draw.circle(mini, (255, 255, 255), (fx, fy), 4)
            pygame.draw.circle(mini, (30,  30,  30),  (fx, fy), 3)

        # Car dot
        raw = env
        while hasattr(raw, "env"):
            raw = raw.env
        car = getattr(raw, "car", None)
        if car is not None:
            car_x, car_y = car.hull.position
            cx, cy = to_mini(car_x, car_y)
            pygame.draw.circle(mini, (220, 20, 20), (cx, cy), 4)

        # Border
        pygame.draw.rect(mini, (15, 158, 117), (0, 0, MINIMAP_SIZE, MINIMAP_SIZE), 2, border_radius=4)

        # Blit to bottom-right
        screen_w, screen_h = self.screen.get_size()
        bx = screen_w  - MINIMAP_SIZE - MINIMAP_MARGIN
        by = screen_h - MINIMAP_SIZE - MINIMAP_MARGIN
        self.screen.blit(mini, (bx, by))

    def _road_polygons(self, env):
        polygons = []
        for road_piece in getattr(env.unwrapped, "road", []):
            for fixture in getattr(road_piece, "fixtures", []):
                vertices = getattr(fixture.shape, "vertices", None)
                if vertices:
                    polygons.append([(float(x), float(y)) for x, y in vertices])
        return polygons


# ─── LEADERBOARD ──────────────────────────────────────────────────────────────
# Structure: list of entries, each with:
#   { "name", "map", "score_type" ("coins"|"time"), "coins"|"lap_time" }

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

def get_board(map_name, score_type):
    """Return entries for a specific map + mode, sorted best first."""
    data = load_leaderboard()
    entries = [e for e in data if e.get("map") == map_name and e.get("score_type") == score_type]
    if score_type == "time":
        entries.sort(key=lambda e: e.get("lap_time", float("inf")))
    else:
        entries.sort(key=leaderboard_score, reverse=True)
    return entries

def add_to_leaderboard(name, map_name, score_type, coins=None, lap_time=None):
    """Add or update a score for a specific map + mode. Keeps best score per name."""
    data = load_leaderboard()
    # Remove old entry for this name/map/mode if it exists and the new score is better
    others = [e for e in data if not (e.get("name") == name and e.get("map") == map_name and e.get("score_type") == score_type)]
    existing = [e for e in data if e.get("name") == name and e.get("map") == map_name and e.get("score_type") == score_type]

    if score_type == "time" and lap_time is not None:
        best_existing = min((e.get("lap_time", float("inf")) for e in existing), default=float("inf"))
        if lap_time < best_existing:
            new_entry = {"name": name, "map": map_name, "score_type": "time", "lap_time": round(lap_time, 3)}
        else:
            new_entry = existing[0] if existing else {"name": name, "map": map_name, "score_type": "time", "lap_time": round(lap_time, 3)}
    else:
        coins = int(coins) if coins is not None else 0
        best_existing = max((e.get("coins", 0) for e in existing), default=-1)
        if coins > best_existing:
            new_entry = {"name": name, "map": map_name, "score_type": "coins", "coins": coins}
        else:
            new_entry = existing[0] if existing else {"name": name, "map": map_name, "score_type": "coins", "coins": coins}

    save_leaderboard(others + [new_entry])
    return get_board(map_name, score_type)


# ─── LAP DETECTION ────────────────────────────────────────────────────────────

def _raw_unwrapped(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e

def track_progress_check(env):
    raw     = _raw_unwrapped(env)
    total   = len(getattr(raw, "track", []))
    visited = getattr(raw, "tile_visited_count", 0)
    return visited, total

def get_start_pos(env):
    raw   = _raw_unwrapped(env)
    track = getattr(raw, "track", [])
    if not track:
        return None
    _alpha, _beta, x, y = track[0]
    return float(x), float(y)

def track_completed_check(env):
    """Lap complete when car returns to tile-0 after doing at least half the track."""
    raw     = _raw_unwrapped(env)
    track   = getattr(raw, "track", [])
    total   = len(track)
    visited = getattr(raw, "tile_visited_count", 0)
    if total == 0 or visited < int(total * LAP_MIN_PROGRESS):
        return False
    car = getattr(raw, "car", None)
    if car is None:
        return False
    start_pos = get_start_pos(env)
    if start_pos is None:
        return False
    car_x, car_y     = car.hull.position
    start_x, start_y = start_pos
    dist = np.sqrt((car_x - start_x) ** 2 + (car_y - start_y) ** 2)
    return dist <= LAP_START_RADIUS


# ─── AI DRIVE ─────────────────────────────────────────────────────────────────

def run_ai(map_name, map_seed, game_mode="coins"):
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

    model     = PPO.load(f"{SAVE_DIR}/{MODEL_NAME}", env=vec_env)
    vec_env.seed(map_seed)
    obs       = vec_env.reset()
    inner_env = vec_env.venv.venv.envs[0]

    start_time    = time.time()
    lap_time      = None
    coins         = 0
    total_tiles   = 0
    final_message = None
    timed_out     = False

    pygame.init()
    mode_label = "Collect Coins" if game_mode == "coins" else "Best Time"
    viewer = RaceViewer(f"Apex AI - AI Drive | {map_name} | {mode_label}")
    clock  = pygame.time.Clock()

    print("AI is driving. Watch the simulation window.\n")

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
            viewer.draw(inner_env.env.render(), inner_env)
            clock.tick(50)

            elapsed            = time.time() - start_time
            coins, total_tiles = track_progress_check(inner_env)
            lap_done_now       = track_completed_check(inner_env)

            print(f"\rAI | {map_name} | {elapsed:.1f}s  tiles: {coins}/{total_tiles}  lap: {lap_done_now}", end="", flush=True)

            if lap_done_now:
                lap_time = time.time() - start_time
                print(f"\nAI completed the lap in {lap_time:.2f}s" if game_mode == "time" else f"\nLap complete. Score: {coins} coins")
                running = False

            elif done[0]:
                final_message = [
                    "Driving Finished", "Goal reached: No",
                    f"Track progress: {coins}/{total_tiles} tiles" if game_mode == "time" else f"Coins collected: {coins}/{total_tiles}",
                    "Complete the full lap to set a time!" if game_mode == "time" else f"{TIME_LIMIT_SECONDS}s rule: more coins wins",
                ]
                print(f"\nEpisode ended ({coins}/{total_tiles} tiles).")
                running = False

            elif game_mode == "coins" and elapsed >= TIME_LIMIT_SECONDS:
                timed_out     = True
                final_message = [
                    "Driving Out Of Time", "Goal reached: No",
                    f"Coins collected: {coins}/{total_tiles}",
                    f"{TIME_LIMIT_SECONDS}s rule: more coins wins",
                ]
                print(f"\nOut of time ({TIME_LIMIT_SECONDS}s).")
                running = False

            elif game_mode == "time" and elapsed >= 40:
                timed_out     = True
                final_message = [
                    "Time's Up!", "AI did not complete the lap in 40s.",
                    f"Track progress: {coins}/{total_tiles} tiles",
                ]
                print("\nAI ran out of time (40s).")
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

    board = get_board(map_name, game_mode)

    if game_mode == "time":
        if lap_time is not None:
            board = add_to_leaderboard("AI", map_name, "time", lap_time=lap_time)
        show_ai_result_time(lap_time, map_name, board)
    else:
        board = add_to_leaderboard("AI", map_name, "coins", coins=coins)
        show_ai_result(coins, map_name, board)


# ─── RESULT SCREENS ───────────────────────────────────────────────────────────

def show_ai_result(ai_coins, map_name, board):
    root = tk.Tk()
    root.title("AI Race Result")
    root.geometry("520x460")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    big_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")

    tk.Label(root, text="AI Race Result", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(24, 4))
    tk.Label(root, text=f"{map_name}  —  Collect Coins", font=sub_font, bg="#0D1117", fg="#64748B").pack()
    tk.Label(root, text=f"AI Coins: {ai_coins}", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=(12, 4))

    humans = [e for e in board if e["name"] != "AI"]
    if humans:
        best = humans[0]
        human_coins = leaderboard_score(best)
        tk.Label(root, text=f"Human Best: {human_coins} coins  ({best['name']})", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)
        if ai_coins > human_coins:
            verdict, color = f"AI WINS by {ai_coins - human_coins} coins", "#0F9E75"
        elif ai_coins == human_coins:
            verdict, color = "Tie!", "#D97706"
        else:
            verdict, color = f"Human still wins by {human_coins - ai_coins} coins", "#D97706"
        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(12, 4))
    else:
        tk.Label(root, text="No human scores on this map yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text=f"Leaderboard — {map_name} — Coins", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(12, 4))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(board[:5]):
        rank  = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(12, 0))
    root.mainloop()


def show_ai_result_time(ai_lap_time, map_name, board):
    root = tk.Tk()
    root.title("AI Race Result — Best Time")
    root.geometry("520x460")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    big_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")

    tk.Label(root, text="AI Race Result", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(24, 4))
    tk.Label(root, text=f"{map_name}  —  Best Time", font=sub_font, bg="#0D1117", fg="#64748B").pack()

    if ai_lap_time is not None:
        tk.Label(root, text=f"AI Lap Time: {ai_lap_time:.2f}s", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=(12, 4))
    else:
        tk.Label(root, text="AI did not complete the lap.", font=big_font, bg="#0D1117", fg="#DC2626").pack(pady=(12, 4))

    humans = [e for e in board if e["name"] != "AI"]
    if humans and ai_lap_time is not None:
        best       = humans[0]
        human_time = best.get("lap_time", float("inf"))
        tk.Label(root, text=f"Human Best: {human_time:.2f}s  ({best['name']})", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)
        if ai_lap_time < human_time:
            verdict, color = f"AI WINS by {human_time - ai_lap_time:.2f}s", "#0F9E75"
        elif ai_lap_time == human_time:
            verdict, color = "Tie!", "#D97706"
        else:
            verdict, color = f"Human still wins by {ai_lap_time - human_time:.2f}s", "#D97706"
        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(12, 4))
    else:
        tk.Label(root, text="No human times on this map yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text=f"Leaderboard — {map_name} — Best Time", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(12, 4))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(board[:5]):
        rank  = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {entry.get('lap_time', 0):.2f}s", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(12, 0))
    root.mainloop()


def show_leaderboard_screen(name, coins, map_name, board):
    root = tk.Tk()
    root.title("Race Finished")
    root.geometry("520x480")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=12, weight="bold")
    row_font   = tkfont.Font(family="Helvetica", size=12)

    tk.Label(root, text="Race Finished",              font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(24, 4))
    tk.Label(root, text=f"{map_name}  —  Collect Coins", font=sub_font, bg="#0D1117", fg="#64748B").pack()
    tk.Label(root, text=f"{name}  —  {coins} coins",  font=sub_font,   bg="#0D1117", fg="#FFFFFF").pack(pady=(8, 16))
    tk.Label(root, text=f"Leaderboard — {map_name} — Coins", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(board[:8]):
        rank   = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and leaderboard_score(entry) == coins
        color  = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins{suffix}", font=row_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=btn_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", activebackground="#334155", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()


def show_time_result_screen(name, lap_time, map_name, board):
    root = tk.Tk()
    root.title("Best Time Result")
    root.geometry("520x480")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=12, weight="bold")
    row_font   = tkfont.Font(family="Helvetica", size=12)

    tk.Label(root, text="Lap Complete!",                  font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(24, 4))
    tk.Label(root, text=f"{map_name}  —  Best Time",      font=sub_font,   bg="#0D1117", fg="#64748B").pack()
    tk.Label(root, text=f"{name}  —  {lap_time:.2f}s",    font=sub_font,   bg="#0D1117", fg="#FFFFFF").pack(pady=(8, 16))
    tk.Label(root, text=f"Leaderboard — {map_name} — Best Time", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(board[:8]):
        rank   = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and abs(entry.get("lap_time", -1) - lap_time) < 0.01
        color  = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {entry.get('lap_time', 0):.2f}s{suffix}", font=row_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=btn_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", activebackground="#334155", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()


# ─── GUI LAUNCHER ─────────────────────────────────────────────────────────────

def launch_gui():
    first_map_name = next(iter(MAP_OPTIONS))
    result = {"name": None, "mode": None, "map_name": first_map_name, "map_seed": MAP_OPTIONS[first_map_name], "game_mode": "coins"}

    root = tk.Tk()
    root.title("Apex AI — Launcher")
    root.geometry("520x600")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=26, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=13, weight="bold")
    label_font = tkfont.Font(family="Helvetica", size=11)

    tk.Label(root, text="Apex AI",                      font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(30, 4))
    tk.Label(root, text="AI Racing — Human vs Machine", font=sub_font,   bg="#0D1117", fg="#64748B").pack(pady=(0, 20))
    tk.Label(root, text="Enter your name:",             font=label_font, bg="#0D1117", fg="#FFFFFF").pack()

    name_var   = tk.StringVar()
    name_entry = tk.Entry(root, textvariable=name_var, font=label_font, width=28,
                          bg="#1E293B", fg="#FFFFFF", insertbackground="#FFFFFF", relief="flat", bd=8)
    name_entry.pack(pady=(6, 20))
    name_entry.focus()

    tk.Label(root, text="Choose map:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()
    map_var  = tk.StringVar(value=first_map_name)
    map_menu = tk.OptionMenu(root, map_var, *MAP_OPTIONS.keys())
    map_menu.config(font=label_font, width=24, bg="#1E293B", fg="#FFFFFF",
                    activebackground="#334155", activeforeground="#FFFFFF", relief="flat", highlightthickness=0)
    map_menu["menu"].config(bg="#1E293B", fg="#FFFFFF", activebackground="#334155")
    map_menu.pack(pady=(6, 16))

    tk.Label(root, text="Game mode:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()
    GAME_MODES    = {"Collect Coins": "coins", "Best Time": "time"}
    game_mode_var = tk.StringVar(value="Collect Coins")
    gm_menu       = tk.OptionMenu(root, game_mode_var, *GAME_MODES.keys())
    gm_menu.config(font=label_font, width=24, bg="#1E293B", fg="#FFFFFF",
                   activebackground="#334155", activeforeground="#FFFFFF", relief="flat", highlightthickness=0)
    gm_menu["menu"].config(bg="#1E293B", fg="#FFFFFF", activebackground="#334155")
    gm_menu.pack(pady=(6, 16))

    error_label = tk.Label(root, text="", font=sub_font, bg="#0D1117", fg="#DC2626")
    error_label.pack()

    def start_human():
        name = name_var.get().strip()
        if not name:
            error_label.config(text="Please enter your name before starting.")
            return
        result["name"]      = name
        result["mode"]      = "human"
        result["map_name"]  = map_var.get()
        result["map_seed"]  = MAP_OPTIONS[map_var.get()]
        result["game_mode"] = GAME_MODES[game_mode_var.get()]
        root.destroy()

    tk.Button(root, text="Drive Manually", font=btn_font, command=start_human,
              bg="#0F9E75", fg="#FFFFFF", activebackground="#0A6E52", activeforeground="#FFFFFF",
              relief="flat", padx=20, pady=10, width=22).pack(pady=(6, 6))

    def start_ai():
        if not os.path.exists(f"{SAVE_DIR}/{MODEL_NAME}.zip"):
            error_label.config(text="No trained model found. Run train.py first.")
            return
        result["mode"]      = "ai"
        result["map_name"]  = map_var.get()
        result["map_seed"]  = MAP_OPTIONS[map_var.get()]
        result["game_mode"] = GAME_MODES[game_mode_var.get()]
        root.destroy()

    tk.Button(root, text="Let AI Drive", font=btn_font, command=start_ai,
              bg="#1E293B", fg="#64748B", activebackground="#1E293B", activeforeground="#64748B",
              relief="flat", padx=20, pady=10, width=22).pack(pady=(0, 12))

    # Show leaderboard preview for selected map+mode
    def refresh_preview(*_):
        for w in preview_frame.winfo_children():
            w.destroy()
        selected_map  = map_var.get()
        selected_mode = GAME_MODES[game_mode_var.get()]
        board = get_board(selected_map, selected_mode)
        mode_label_str = "Coins" if selected_mode == "coins" else "Best Time"
        tk.Label(preview_frame, text=f"{selected_map} — {mode_label_str}",
                 font=label_font, bg="#0D1117", fg="#D97706").pack()
        if board:
            ranks = ["1st", "2nd", "3rd"]
            for i, entry in enumerate(board[:3]):
                if selected_mode == "time":
                    score_str = f"{entry.get('lap_time', 0):.2f}s"
                else:
                    score_str = f"{leaderboard_score(entry)} coins"
                tk.Label(preview_frame, text=f"{ranks[i]}  {entry['name']}  —  {score_str}",
                         font=sub_font, bg="#0D1117", fg="#FFFFFF").pack()
        else:
            tk.Label(preview_frame, text="No scores yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack()

    preview_frame = tk.Frame(root, bg="#0D1117")
    preview_frame.pack(pady=(0, 8))
    map_var.trace_add("write", refresh_preview)
    game_mode_var.trace_add("write", refresh_preview)
    refresh_preview()

    root.mainloop()
    return result


# ─── MAIN GAME (human) ────────────────────────────────────────────────────────

def run_game(player_name, map_name, map_seed, game_mode="coins"):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        max_episode_steps=MAX_RACE_STEPS,
        lap_complete_percent=LAP_COMPLETE_PERCENT,
    )
    obs, info = env.reset(seed=map_seed)

    pygame.init()
    mode_label = "Collect Coins" if game_mode == "coins" else "Best Time"
    viewer = RaceViewer(f"Apex AI - Manual Drive | {map_name} | {mode_label}")
    clock  = pygame.time.Clock()

    print(f"\nWelcome {player_name}. {map_name} | {mode_label}")
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
                    "Time's Up!", f"{TIME_LIMIT_SECONDS}s limit reached.",
                    f"Coins collected: {coins}/{total_t}",
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
                    f"Coins: {visited}/{total_t}" if game_mode == "coins" else f"Progress: {visited}/{total_t} tiles",
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


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        result = launch_gui()

        if result["mode"] is None:
            print("No mode selected. Exiting.")
            break

        map_name  = result["map_name"]
        map_seed  = result["map_seed"]
        game_mode = result.get("game_mode", "coins")

        if result["mode"] == "human":
            player_name = result["name"]
            score_type, score_value = run_game(player_name, map_name, map_seed, game_mode)

            if score_type == "coins" and score_value is not None:
                board = add_to_leaderboard(player_name, map_name, "coins", coins=score_value)
                show_leaderboard_screen(player_name, score_value, map_name, board)
            elif score_type == "time" and score_value is not None:
                board = add_to_leaderboard(player_name, map_name, "time", lap_time=score_value)
                show_time_result_screen(player_name, score_value, map_name, board)
            else:
                print("No score recorded — returning to menu.")

        elif result["mode"] == "ai":
            run_ai(map_name, map_seed, game_mode)