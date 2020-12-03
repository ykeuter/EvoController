from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsWorld:
    def __init__(self, file_name=None):
        self.env = None
        self.behavior_name = None
        self.file_name = file_name

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(
            width=84,
            height=84,
            quality_level=0,
            time_scale=20,
            target_frame_rate=-1,
            # capture_frame_rate=60
        )
        self.env = UnityEnvironment(
            file_name=self.file_name,
            side_channels=[channel],
            # no_graphics=True,
        )
        # Start interacting with the environment.
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.observe()
        return obs

    def step(self, action):
        self.env.set_action_for_agent(self.behavior_name, 0, action)
        self.env.step()
        return self.observe()

    def observe(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        steps = terminal_steps if terminal_steps else decision_steps
        obs = steps.obs[0][0, :]
        reward = steps.reward[0]
        done = len(terminal_steps) != 0
        info = {}
        return obs, reward, done, info
