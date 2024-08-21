import gym
import os
from stable_baselines3 import PPO
import time  # Import time module for adding delay

# Optional: Initialize wandb if you want to log the performance
import wandb

# Create the environment with rendering
env = gym.make("CarRacing-v2", render_mode="human")

# Load the trained PPO model
model_file = "ppo_carracing.zip"
if os.path.exists(model_file):
    model = PPO.load(model_file)
    print("Loaded PPO model from file.")
else:
    raise FileNotFoundError(f"Model file {model_file} not found.")

# Run the model
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()  # Enable rendering
    time.sleep(0.01)  # Add a small delay to make rendering visible

env.close()
print(f"Total reward: {total_reward}")
