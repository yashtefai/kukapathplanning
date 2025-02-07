import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
from gym import spaces
import os


class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Connect to PyBullet in GUI mode for visualization
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the robot model
        urdf_path = "kuka_iiwa/model.urdf"
        if not os.path.exists(os.path.join(pybullet_data.getDataPath(), urdf_path)):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        
        if self.num_joints == 0:
            raise ValueError("Error: Robot URDF failed to load correctly, no joints detected!")

        print(f"Loaded robot with {self.num_joints} joints.")

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints + 3,), dtype=np.float32)

        # Define square path waypoints (Fixed Height)
        self.square_path = [
            np.array([0.4, 0.4, 0.2]),  # A
            np.array([0.4, -0.4, 0.2]), # B
            np.array([-0.4, -0.4, 0.2]),# C
            np.array([-0.4, 0.4, 0.2])  # D
        ]
        
        self.waypoint_index = 0  # Start at the first waypoint

        # Set camera position for better view
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

    def reset(self):
        """ Reset environment for a new episode """
        print(f"Resetting environment... Number of joints: {self.num_joints}")

        if self.num_joints == 0:
            raise ValueError("Reset failed: No joints detected in the robot!")

        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, np.random.uniform(-0.5, 0.5))

        self.waypoint_index = 0  # Start at the first waypoint
        return self._get_observation()

    def step(self, action):
        """ Apply action and compute next state, reward, and done flag """
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.VELOCITY_CONTROL, targetVelocity=action[i])

        p.stepSimulation()
        time.sleep(0.005)  # Slow down for visualization

        obs = self._get_observation()
        reward, done = self._compute_reward(obs)

        return obs, reward, done, {}

    def _get_observation(self):
        """ Get joint positions + next goal position """
        if self.num_joints == 0:
            return np.zeros(self.observation_space.shape)

        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        next_goal = self.square_path[self.waypoint_index]  # Next waypoint in path
        return np.concatenate([joint_states, next_goal])

    def _compute_reward(self, obs):
        """ Compute reward based on distance to the next waypoint """
        if self.num_joints == 0:
            return -1.0, False

        ee_index = self.num_joints - 1  
        ee_pos = np.array(p.getLinkState(self.robot, ee_index)[0])  # End effector position
        target_pos = self.square_path[self.waypoint_index]

        distance = np.linalg.norm(ee_pos - target_pos)

        # Reward based on distance to target
        reward = -distance  # Closer is better

        # Bonus for reaching the waypoint
        if distance < 0.05:
            reward += 10  # Big reward for reaching target
            print(f"Reached waypoint {self.waypoint_index + 1} / 4")
            self.waypoint_index = (self.waypoint_index + 1) % len(self.square_path)  # Move to next target

        return reward, False  # No episode termination

    def render(self, mode="human"):
        """ Already using PyBullet GUI, so nothing extra needed """
        pass

    def close(self):
        """ Clean up PyBullet simulation """
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()


# Run a simple test loop
if __name__ == "__main__":
    env = RobotEnv()
    obs = env.reset()

    for _ in range(500):  # Run 500 steps
        action = np.random.uniform(-1, 1, env.action_space.shape)  # Random actions
        obs, reward, done, _ = env.step(action)
    
    env.close()
