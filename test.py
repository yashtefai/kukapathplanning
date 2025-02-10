import gym
import time
from stable_baselines3 import PPO
from robot_env import RobotEnv
import pybullet as p

# Load environment and trained model
env = RobotEnv()
model = PPO.load("ppo_robot")

# Enable visualization
env.render()  # Make sure this is called

obs = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    
    p.stepSimulation()  # Ensure simulation steps forward
    time.sleep(0.1)

    if done:
        print("Goal reached!")
        break

env.close()