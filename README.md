# AI_Robotics_Project
AI in Robotics Project UTS 2026 
Step 1 — Make sure you have Python 3.11 installed
Download from: https://www.python.org/downloads/release/python-3119/
During install, tick "Add to PATH"

Step 2 — Open terminal and navigate to the project folder

Step 3 — Install all dependencies
bashpy -3.11 -m pip install gymnasium[box2d] pygame stable-baselines3 numpy

Step 4 — Drive manually first to set a human lap time
bashpy -3.11 run_car.py

Enter your name in the GUI
Click Drive Manually
Use W A S D to drive
Timer starts when you first press W
Lap time saves automatically when the episode ends
Do this a few times to get a good benchmark time


Step 5 — Train the AI
bashpy -3.11 train.py

Leave this running — it takes a few hours
You will see training progress printing in the terminal
Press Ctrl+C anytime to stop early — progress is saved
Model saves automatically to the models/ folder


Step 6 — Watch the AI drive and compare times
bashpy -3.11 run_car.py

Open the GUI again
Click Let AI Drive
Watch the simulation window
Result screen shows AI time vs your best human time
Leaderboard updates automatically

Progress:
- WASD Control Done
- GUI Done
- Train file Done (Not trained yet)