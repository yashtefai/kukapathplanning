import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Connect to PyBullet in DIRECT mode for faster training
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the robot and environment
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints + 3,), dtype=np.float32)

        # Define square path waypoints
        self.square_path = [
            [0.5, 0.5, 0.2],
            [0.5, -0.5, 0.2],
            [-0.5, -0.5, 0.2],
            [-0.5, 0.5, 0.2]
        ]
        self.current_waypoint = 0

    def reset(self):
        """ Reset environment for a new episode """
        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, np.random.uniform(-0.5, 0.5))

        self.current_waypoint = 0
        return self._get_observation()

    def step(self, action):
        """ Apply action and compute next state, reward, and done flag """
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.VELOCITY_CONTROL, targetVelocity=action[i])
        
        p.stepSimulation()

        obs = self._get_observation()
        reward, done = self._compute_reward(obs)

        return obs, reward, done, {}

    def _get_observation(self):
        """ Get joint positions + target waypoint """
        joint_states = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        return np.concatenate([joint_states, self.square_path[self.current_waypoint]])

    def _compute_reward(self, obs):
        """ Compute reward based on distance to current waypoint """
        ee_pos = np.array(p.getLinkState(self.robot, self.num_joints - 1)[0])  # End effector position
        distance = np.linalg.norm(ee_pos - np.array(self.square_path[self.current_waypoint]))

        reward = -distance  # Reward is negative distance to waypoint
        if distance < 0.05:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.square_path)  # Move to next waypoint

        done = False  # The task never ends in training
        return reward, done

    def render(self, mode="human"):
         """ Enable PyBullet GUI for visualization """
         if mode == "human":
            p.disconnect()  # Disconnect any previous connections
            self.physicsClient = p.connect(p.GUI)  # Start GUI mode
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

    def close(self):
        """ Clean up PyBullet simulation """
        p.disconnect()
