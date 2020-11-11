from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsWorld:
    def __init__(self):
        self.env = None
        self.name = None

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(
            width=84,
            height=84,
            quality_level=5,
            time_scale=20,
            target_frame_rate=-1,
            capture_frame_rate=60)
        self.env = UnityEnvironment(side_channels=[channel])
        # Start interacting with the environment.
        self.env.reset()
        self.name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def step(self, action):
        self.env.set_action_for_agent(self.name, 0, action)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.name)
        steps = decision_steps | terminal_steps
        obs = steps.obs[0]
        reward = steps.reward[0]
        done = len(decision_steps) == 0
        info = {}
        return obs, reward, done, info
