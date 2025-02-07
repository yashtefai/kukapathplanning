import gym
from stable_baselines3 import PPO
from robot_env import RobotEnv

# Load environment and trained model
env = RobotEnv()
model = PPO.load("ppo_robot")

# Enable visualization
env.render()

obs = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    if done:
        print("Goal reached!")
        break

env.close()
