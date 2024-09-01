import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import torch as th
import torch.nn as nn
import numpy as np
import wandb
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ELU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ELU(),  
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),  
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, features_dim),
            nn.ELU(), 
            nn.Dropout(p=0.5)
        )

    def forward(self, observations):
        return self.cnn(observations)


class SpeedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SpeedRewardWrapper, self).__init__(env)
        self.start_time = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.start_time = (
            self.env.unwrapped.t
        )  
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            end_time = self.env.unwrapped.t
            time_taken = end_time - self.start_time
            reward += max(
                0, 1000 - time_taken
            ) 

       
        speed = self.env.unwrapped.car.hull.linearVelocity.length
        off_track = self.env.unwrapped.car.on_grass

        
        speed_bonus = speed * 0.1
        reward += speed_bonus

        # Penalty for going off-track
        if off_track:
            reward -= 10

        return obs, reward, done, info



wandb.init(
    entity="WANDB ACCOUNT HERE",
    project="",
    name="",
    #id="",
    resume="",
)


env = make_vec_env("CarRacing-v2", n_envs=2)


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=dict(pi=[1024, 1024, 1024, 1024, 1024], vf=[1024, 1024, 1024, 1024, 1024]),
)
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,  
    n_steps=2048,  
    batch_size=1024,  
    n_epochs=30, 
    gamma=0.99,  
    gae_lambda=0.95,  
    clip_range=0.1, 
    verbose=1,
)

# Check if a saved model exists
model_file = "ppo_carracing.zip"
if os.path.exists(model_file):
    model = PPO.load(model_file, env=env)
    print("Loaded PPO model from file.")


timesteps_per_update = 1000
fps = 95 

i=1
while True:
    print(f"Training iteration {i}...")
    model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)

    
    ep_lengths = [ep_info["l"] for ep_info in model.ep_info_buffer]
    ep_rewards = [ep_info["r"] for ep_info in model.ep_info_buffer]
    ep_len_mean = np.mean(ep_lengths) if ep_lengths else 0
    ep_rew_mean = np.mean(ep_rewards) if ep_rewards else 0
    time_elapsed = model.num_timesteps / fps
    total_timesteps = model.num_timesteps

    
    wandb.log(
        {
            "rollout/ep_len_mean": ep_len_mean,  
            "rollout/ep_rew_mean": ep_rew_mean, 
            "time/iterations": i, 
            "time/time_elapsed": time_elapsed,  
            "time/total_timesteps": total_timesteps,  
            "params/learning_rate": model.learning_rate,  
        }
    )

    
    model.save(model_file)
    print("Model saved.")
    i+=1

print("Training complete.")
