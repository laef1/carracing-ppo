import gym
import os
from stable_baselines3 import PPO
import time 


import wandb


env = gym.make("CarRacing-v2", render_mode="human")


model_file = "ppo_carracing.zip"
if os.path.exists(model_file):
    model = PPO.load(model_file)
    print("Loaded PPO model from file.")
else:
    raise FileNotFoundError(f"Model file {model_file} not found.")


obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()  
    time.sleep(0.01)  

env.close()
print(f"Total reward: {total_reward}")
