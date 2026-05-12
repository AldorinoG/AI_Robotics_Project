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

SAVE_DIR         = "./models"
MODEL_NAME       = "racenet_cnn_ppo_legacy_final"
LEADERBOARD_FILE = "leaderboard.json"
VIEWER_SIZE      = (900, 700)
MAX_RACE_STEPS   = 100000
TIME_LIMIT_SECONDS = 30
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

        title_font = pygame.font.SysFont("Helvetica", 30, bold=True)
        text_font = pygame.font.SysFont("Helvetica", 20)
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

        car = getattr(env.unwrapped, "car", None)
        if car is not None:
            car_x, car_y = car.hull.position
            car_angle = -car.hull.angle
            cx, cy = self._world_to_screen(car_x, car_y)
            size = max(8, int(self.map_zoom * 1.4))
            car_points = [
                (cx + np.cos(car_angle) * size, cy + np.sin(car_angle) * size),
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
        self.map_zoom = min(screen_w / map_w, screen_h / map_h) * 0.9
        self.map_pan_x = screen_w / 2 - ((min_x + max_x) / 2) * self.map_zoom
        self.map_pan_y = screen_h / 2 + ((min_y + max_y) / 2) * self.map_zoom

    def _world_to_screen(self, x, y):
        return (
            int(x * self.map_zoom + self.map_pan_x),
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
        road_tiles = getattr(env.unwrapped, "road", [])
        coin_radius = int(np.clip(self.map_zoom * 0.45, 3, 8))

        for index, (_alpha, _beta, x, y) in enumerate(track):
            road_tile = road_tiles[index] if index < len(road_tiles) else None
            collected = bool(getattr(road_tile, "road_visited", False))
            sx, sy = self._world_to_screen(x, y)
            color = (22, 210, 90) if collected else (245, 190, 35)
            outline = (4, 110, 48) if collected else (140, 92, 8)
            pygame.draw.circle(self.screen, outline, (sx, sy), coin_radius + 1)
            pygame.draw.circle(self.screen, color, (sx, sy), coin_radius)

#LEADERBOARD

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


def add_to_leaderboard(name, coins):
    data = load_leaderboard()
    coins = int(coins)
    best_by_name = {}

    for entry in data:
        entry_name = entry["name"]
        if entry_name not in best_by_name or leaderboard_score(entry) > leaderboard_score(best_by_name[entry_name]):
            best_by_name[entry_name] = {"name": entry_name, "coins": leaderboard_score(entry)}

    if name not in best_by_name or coins > leaderboard_score(best_by_name[name]):
        best_by_name[name] = {"name": name, "coins": coins}

    leaderboard = list(best_by_name.values())
    leaderboard.sort(key=leaderboard_score, reverse=True)
    save_leaderboard(leaderboard)
    return leaderboard


def track_completed(env, info=None):
    if info and info.get("track_completed", False):
        return True

    total_tiles = len(getattr(env.unwrapped, "track", []))
    visited_tiles = getattr(env.unwrapped, "tile_visited_count", 0)
    return total_tiles > 0 and visited_tiles >= total_tiles


def track_progress(env, info=None):
    if info:
        visited_tiles = info.get("track_tiles_visited")
        total_tiles = info.get("track_tiles_total")
        if visited_tiles is not None and total_tiles:
            return visited_tiles, total_tiles

    total_tiles = len(getattr(env.unwrapped, "track", []))
    visited_tiles = getattr(env.unwrapped, "tile_visited_count", 0)
    return visited_tiles, total_tiles

#AI DRIVE

def run_ai(map_seed):
    print("Loading trained AI model...")

    def make_env():
        def _init():
            return make_car_racing_env(
                render_mode="rgb_array",
                max_episode_steps=MAX_RACE_STEPS,
                legacy_preprocessing=True,
                terminate_on_no_reward=False,
            )
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(f"{SAVE_DIR}/{MODEL_NAME}", env=env)

    env.seed(map_seed)
    obs = env.reset()
    start_time = time.time()
    lap_time = None
    coins = 0
    final_message = None
    timed_out = False
    pygame.init()
    viewer = RaceViewer("Apex AI - AI Drive")
    clock = pygame.time.Clock()

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
            obs, reward, done, info = env.step(action)
            viewer.draw(env.envs[0].render(), env.envs[0])
            clock.tick(50)

            elapsed = time.time() - start_time
            print(f"\rAI time: {elapsed:.1f}s", end="", flush=True)
            coins, total_tiles = track_progress(env.envs[0], info[0])

            if done[0]:
                if info[0].get("track_completed", False):
                    lap_time = time.time() - start_time
                else:
                    final_message = [
                        "Driving Finished",
                        "Goal reached: No",
                        f"Coins collected: {coins}/{total_tiles}",
                        "30s rule: more coins wins",
                    ]
                    print(
                        f"\nAI episode ended before completing the track "
                        f"({coins}/{total_tiles} coins)."
                    )
                running = False

            elif elapsed >= TIME_LIMIT_SECONDS:
                timed_out = True
                goal_reached = track_completed(env.envs[0], info[0])
                final_message = [
                    "Driving Out Of Time",
                    f"Goal reached: {'Yes' if goal_reached else 'No'}",
                    f"Coins collected: {coins}/{total_tiles}",
                    "30s rule: more coins wins",
                ]
                print(f"\nOut of time. AI reached the {TIME_LIMIT_SECONDS}s limit.")
                running = False
    except KeyboardInterrupt:
        env.close()
        pygame.quit()
        print("\nAI run stopped.")
        return

    if final_message is not None:
        viewer.wait_for_home(env.envs[0].render(), env.envs[0], final_message)

    env.close()
    pygame.quit()

    print(f"\nAI score: {coins} coins")
    if lap_time is not None:
        print(f"AI finished the track in {round(lap_time, 2)}s")

    leaderboard = load_leaderboard()
    humans = [e for e in leaderboard if e["name"] != "AI"]
    human_best = humans[0] if humans else None

    data = add_to_leaderboard("AI", coins)

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
            diff = ai_coins - human_coins
            verdict = f"AI WINS by {diff} coins"
            color = "#0F9E75"
        elif ai_coins == human_coins:
            verdict = "Tie score"
            color = "#D97706"
        else:
            diff = human_coins - ai_coins
            verdict = f"Human still wins by {diff} coins"
            color = "#D97706"

        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(20, 4))
    else:
        tk.Label(root, text="No human times recorded yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text="Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_scores[:5]):
        rank = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()

#GUI LAUNCHER

def launch_gui():
    first_map_name = next(iter(MAP_OPTIONS))
    result = {"name": None, "mode": None, "map_seed": MAP_OPTIONS[first_map_name]}

    root = tk.Tk()
    root.title("Apex AI — Launcher")
    root.geometry("500x480")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=26, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    btn_font   = tkfont.Font(family="Helvetica", size=13, weight="bold")
    label_font = tkfont.Font(family="Helvetica", size=11)

    tk.Label(root, text="Apex AI", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(30, 4))
    tk.Label(root, text="AI Racing — Human vs Machine", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=(0, 24))

    tk.Label(root, text="Enter your name:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()

    name_var = tk.StringVar()
    name_entry = tk.Entry(
        root, textvariable=name_var,
        font=label_font, width=28,
        bg="#1E293B", fg="#FFFFFF",
        insertbackground="#FFFFFF",
        relief="flat", bd=8
    )
    name_entry.pack(pady=(6, 24))
    name_entry.focus()

    tk.Label(root, text="Choose map:", font=label_font, bg="#0D1117", fg="#FFFFFF").pack()

    map_var = tk.StringVar(value=first_map_name)
    map_menu = tk.OptionMenu(root, map_var, *MAP_OPTIONS.keys())
    map_menu.config(
        font=label_font,
        width=24,
        bg="#1E293B",
        fg="#FFFFFF",
        activebackground="#334155",
        activeforeground="#FFFFFF",
        relief="flat",
        highlightthickness=0,
    )
    map_menu["menu"].config(bg="#1E293B", fg="#FFFFFF", activebackground="#334155")
    map_menu.pack(pady=(6, 18))

    error_label = tk.Label(root, text="", font=sub_font, bg="#0D1117", fg="#DC2626")
    error_label.pack()

    def start_human():
        name = name_var.get().strip()
        if not name:
            error_label.config(text="Please enter your name before starting.")
            return
        result["name"] = name
        result["mode"] = "human"
        result["map_seed"] = MAP_OPTIONS[map_var.get()]
        root.destroy()

    tk.Button(
        root, text="Drive Manually",
        font=btn_font, command=start_human,
        bg="#0F9E75", fg="#FFFFFF",
        activebackground="#0A6E52", activeforeground="#FFFFFF",
        relief="flat", padx=20, pady=10, width=22
    ).pack(pady=(8, 8))

    def start_ai():
        model_path = f"{SAVE_DIR}/{MODEL_NAME}"
        if not os.path.exists(model_path + ".zip"):
            error_label.config(text="No trained model found. Run train.py first.")
            return
        result["mode"] = "ai"
        result["map_seed"] = MAP_OPTIONS[map_var.get()]
        root.destroy()

    tk.Button(
        root, text="Let AI Drive",
        font=btn_font, command=start_ai,
        bg="#1E293B", fg="#64748B",
        activebackground="#1E293B", activeforeground="#64748B",
        relief="flat", padx=20, pady=10, width=22
    ).pack(pady=(0, 8))

    data = load_leaderboard()
    if data:
        data.sort(key=leaderboard_score, reverse=True)
        tk.Label(root, text="Top Coin Scores", font=label_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
        ranks = ["1st", "2nd", "3rd"]
        for i, entry in enumerate(data[:3]):
            rank = ranks[i]
            tk.Label(
                root,
                text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins",
                font=sub_font, bg="#0D1117", fg="#FFFFFF"
            ).pack()

    root.mainloop()
    return result

#LEADERBOARD SCREEN

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

    tk.Label(root, text="Race Finished", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 4))
    tk.Label(root, text=f"{name}  —  {coins} coins", font=sub_font, bg="#0D1117", fg="#FFFFFF").pack(pady=(0, 20))

    tk.Label(root, text="Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_scores[:8]):
        rank = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and leaderboard_score(entry) == coins
        color = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(
            root,
            text=f"{rank}  {entry['name']}  —  {leaderboard_score(entry)} coins{suffix}",
            font=row_font, bg="#0D1117", fg=color
        ).pack(pady=1)

    tk.Button(
        root, text="Close",
        font=btn_font, command=root.destroy,
        bg="#1E293B", fg="#FFFFFF",
        activebackground="#334155",
        relief="flat", padx=16, pady=8
    ).pack(pady=(20, 0))

    root.mainloop()

# MAIN GAME 

def run_game(player_name, map_seed):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        max_episode_steps=MAX_RACE_STEPS,
    )
    obs, info = env.reset(seed=map_seed)

    pygame.init()
    viewer = RaceViewer("Apex AI - Manual Drive")
    clock = pygame.time.Clock()

    print(f"\nWelcome {player_name}.")
    print("Controls: W = gas | S = brake | A = left | D = right | Q = quit")
    print("Timer starts when you first press W.\n")

    start_time = None
    lap_done   = False
    lap_time   = None
    coins      = 0
    timed_out  = False
    final_message = None

    running = True
    while running:
        action = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        for event in pygame.event.get():
            if not viewer.handle_event(event):
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            action[0] = -1.0
        if keys[pygame.K_d]:
            action[0] = 1.0
        if keys[pygame.K_w]:
            action[1] = 1.0
            if start_time is None:
                start_time = time.time()
                print("Timer started.")
        if keys[pygame.K_s]:
            action[2] = 0.8
        if keys[pygame.K_q]:
            running = False

        obs, reward, terminated, truncated, info = env.step(action)
        viewer.draw(env.render(), env)
        clock.tick(50)

        if start_time is not None and not lap_done:
            elapsed = time.time() - start_time
            print(f"\r{elapsed:.1f}s", end="", flush=True)
            coins, total_tiles = track_progress(env, info)

            if track_completed(env, info):
                lap_time = time.time() - start_time
                lap_done = True
                print(f"\nLap complete. Score: {coins} coins")
                running = False
            elif elapsed >= TIME_LIMIT_SECONDS:
                timed_out = True
                final_message = [
                    "Driving Out Of Time",
                    "Goal reached: No",
                    f"Coins collected: {coins}/{total_tiles}",
                    "30s rule: more coins wins",
                ]
                print(f"\nOut of time. You reached the {TIME_LIMIT_SECONDS}s limit.")
                running = False

        if running and (terminated or truncated) and not lap_done:
            if start_time is not None:
                if track_completed(env, info):
                    lap_time = time.time() - start_time
                    lap_done = True
                    coins, _total_tiles = track_progress(env, info)
                    print(f"\nLap complete. Score: {coins} coins")
                else:
                    visited_tiles, total_tiles = track_progress(env, info)
                    coins = visited_tiles
                    final_message = [
                        "Driving Finished",
                        "Goal reached: No",
                        f"Coins collected: {visited_tiles}/{total_tiles}",
                        "30s rule: more coins wins",
                    ]
                    print(
                        f"\nEpisode ended before completing the track "
                        f"({visited_tiles}/{total_tiles} tiles)."
                    )
                running = False
            else:
                obs, info = env.reset()

    if final_message is not None:
        viewer.wait_for_home(env.render(), env, final_message)

    env.close()
    pygame.quit()
    if timed_out:
        print("Returning to menu.")
    return coins if start_time is not None else None

# ENTRY POINT

if __name__ == "__main__":
    while True:
        result = launch_gui()

        if result["mode"] is None:
            print("No mode selected. Exiting.")
            break

        if result["mode"] == "human":
            player_name = result["name"]
            coins = run_game(player_name, result["map_seed"])

            if coins is not None:
                all_scores = add_to_leaderboard(player_name, coins)
                show_leaderboard_screen(player_name, coins, all_scores)
            else:
                print("No coin score recorded — returning to menu.")

        elif result["mode"] == "ai":
            run_ai(result["map_seed"])
