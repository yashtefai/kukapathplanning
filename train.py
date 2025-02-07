import gym
from stable_baselines3 import PPO
from robot_env import RobotEnv  # Import our custom environment

# Create environment
env = RobotEnv()

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.99, n_steps=2048, batch_size=64)

# Train the model
print("Training started...")
model.learn(total_timesteps=100000)
print("Training finished!")

# Save model
model.save("ppo_robot")
print("Model saved!")
env.close()
