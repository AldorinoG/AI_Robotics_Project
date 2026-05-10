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

#LEADERBOARD

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []

def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_to_leaderboard(name, lap_time):
    data = load_leaderboard()
    data.append({"name": name, "time": round(lap_time, 2)})
    data.sort(key=lambda x: x["time"])
    save_leaderboard(data)
    return data

#AI DRIVE

def run_ai():
    print("Loading trained AI model...")

    def make_env():
        def _init():
            return make_car_racing_env(
                render_mode="human",
                legacy_preprocessing=True,
                terminate_on_no_reward=False,
            )
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(f"{SAVE_DIR}/{MODEL_NAME}", env=env)

    obs = env.reset()
    start_time = time.time()
    lap_time = None
    attempts = 1

    print("AI is driving.")
    print("Watch the simulation window.\n")

    running = True
    try:
        while running:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            elapsed = time.time() - start_time
            print(f"\rAI attempt {attempts} time: {elapsed:.1f}s", end="", flush=True)

            if done[0]:
                if not info[0].get("lap_finished", False):
                    attempts += 1
                    print("\nAI went off track. Resetting and trying again.\n")
                    obs = env.reset()
                    start_time = time.time()
                    continue

                lap_time = time.time() - start_time
                running = False
    except KeyboardInterrupt:
        env.close()
        print("\nAI run stopped.")
        return

    env.close()
    print(f"\nAI finished. Lap time: {round(lap_time, 2)}s")

    leaderboard = load_leaderboard()
    humans = [e for e in leaderboard if e["name"] != "AI"]
    human_best = humans[0] if humans else None

    data = load_leaderboard()
    data.append({"name": "AI", "time": round(lap_time, 2)})
    data.sort(key=lambda x: x["time"])
    save_leaderboard(data)

    show_ai_result(lap_time, human_best, data)


def show_ai_result(ai_time, human_best, all_times):
    root = tk.Tk()
    root.title("AI Race Result")
    root.geometry("500x420")
    root.resizable(False, False)
    root.configure(bg="#0D1117")

    title_font = tkfont.Font(family="Helvetica", size=22, weight="bold")
    sub_font   = tkfont.Font(family="Helvetica", size=11)
    big_font   = tkfont.Font(family="Helvetica", size=14, weight="bold")

    tk.Label(root, text="AI Race Result", font=title_font, bg="#0D1117", fg="#0F9E75").pack(pady=(28, 16))
    tk.Label(root, text=f"AI Lap Time:   {round(ai_time, 2)}s", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)

    if human_best:
        tk.Label(root, text=f"Human Best:   {human_best['time']}s  ({human_best['name']})", font=big_font, bg="#0D1117", fg="#FFFFFF").pack(pady=4)

        if ai_time < human_best["time"]:
            diff = round(human_best["time"] - ai_time, 2)
            verdict = f"AI WINS by {diff}s"
            color = "#0F9E75"
        else:
            diff = round(ai_time - human_best["time"], 2)
            verdict = f"Human still wins by {diff}s — keep training."
            color = "#D97706"

        tk.Label(root, text=verdict, font=big_font, bg="#0D1117", fg=color).pack(pady=(20, 4))
    else:
        tk.Label(root, text="No human times recorded yet.", font=sub_font, bg="#0D1117", fg="#64748B").pack(pady=8)

    tk.Label(root, text="Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_times[:5]):
        rank = ranks[i] if i < 3 else f"{i+1}th"
        color = "#0F9E75" if entry["name"] == "AI" else "#FFFFFF"
        tk.Label(root, text=f"{rank}  {entry['name']}  —  {entry['time']}s", font=sub_font, bg="#0D1117", fg=color).pack(pady=1)

    tk.Button(root, text="Close", font=sub_font, command=root.destroy, bg="#1E293B", fg="#FFFFFF", relief="flat", padx=16, pady=8).pack(pady=(16, 0))
    root.mainloop()

#GUI LAUNCHER

def launch_gui():
    result = {"name": None, "mode": None}

    root = tk.Tk()
    root.title("Apex AI — Launcher")
    root.geometry("500x420")
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

    error_label = tk.Label(root, text="", font=sub_font, bg="#0D1117", fg="#DC2626")
    error_label.pack()

    def start_human():
        name = name_var.get().strip()
        if not name:
            error_label.config(text="Please enter your name before starting.")
            return
        result["name"] = name
        result["mode"] = "human"
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
        root.destroy()
        run_ai()

    tk.Button(
        root, text="Let AI Drive",
        font=btn_font, command=start_ai,
        bg="#1E293B", fg="#64748B",
        activebackground="#1E293B", activeforeground="#64748B",
        relief="flat", padx=20, pady=10, width=22
    ).pack(pady=(0, 8))

    data = load_leaderboard()
    if data:
        tk.Label(root, text="Top Times", font=label_font, bg="#0D1117", fg="#D97706").pack(pady=(16, 4))
        ranks = ["1st", "2nd", "3rd"]
        for i, entry in enumerate(data[:3]):
            rank = ranks[i]
            tk.Label(
                root,
                text=f"{rank}  {entry['name']}  —  {entry['time']}s",
                font=sub_font, bg="#0D1117", fg="#FFFFFF"
            ).pack()

    root.mainloop()
    return result

#LEADERBOARD SCREEN

def show_leaderboard_screen(name, lap_time, all_times):
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
    tk.Label(root, text=f"{name}  —  {round(lap_time, 2)}s", font=sub_font, bg="#0D1117", fg="#FFFFFF").pack(pady=(0, 20))

    tk.Label(root, text="Leaderboard", font=sub_font, bg="#0D1117", fg="#D97706").pack(pady=(0, 8))

    ranks = ["1st", "2nd", "3rd"]
    for i, entry in enumerate(all_times[:8]):
        rank = ranks[i] if i < 3 else f"{i+1}th"
        is_you = entry["name"] == name and round(entry["time"], 2) == round(lap_time, 2)
        color = "#0F9E75" if is_you else "#FFFFFF"
        suffix = "  <- you" if is_you else ""
        tk.Label(
            root,
            text=f"{rank}  {entry['name']}  —  {entry['time']}s{suffix}",
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

def run_game(player_name):
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, info = env.reset()

    pygame.init()

    print(f"\nWelcome {player_name}.")
    print("Controls: W = gas | S = brake | A = left | D = right | Q = quit")
    print("Timer starts when you first press W.\n")

    start_time = None
    lap_done   = False
    lap_time   = None

    running = True
    while running:
        action = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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

        if start_time is not None and not lap_done:
            elapsed = time.time() - start_time
            print(f"\r{elapsed:.1f}s", end="", flush=True)

        if (terminated or truncated) and not lap_done:
            if start_time is not None:
                lap_time = time.time() - start_time
                lap_done = True
                running  = False
                print(f"\nLap complete. Time: {round(lap_time, 2)}s")
            else:
                obs, info = env.reset()

    env.close()
    pygame.quit()
    return lap_time

# ENTRY POINT

if __name__ == "__main__":
    result = launch_gui()

    if result["mode"] is None:
        print("No mode selected. Exiting.")

    elif result["mode"] == "human":
        player_name = result["name"]
        lap_time = run_game(player_name)

        if lap_time is not None:
            all_times = add_to_leaderboard(player_name, lap_time)
            show_leaderboard_screen(player_name, lap_time, all_times)
        else:
            print("No lap time recorded — you quit before finishing.")

    elif result["mode"] == "ai":
        pass
