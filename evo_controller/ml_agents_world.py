from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsWorld:
    def __init__(self, file_name=None, training=True,
                 num_agents=1, num_inputs=1, angle=0):
        self.env = None
        self.behavior_name = None
        self.file_name = file_name
        self.training = training
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.states = np.empty([num_agents, num_inputs])
        self.num_died = 0
        self.agent_id_to_index = None
        self.last_idx = None
        self.angle = angle

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(
            width=84 if self.training else 512,
            height=84 if self.training else 512,
            quality_level=0,
            time_scale=100 if self.training else 1,
            target_frame_rate=-1,
            # capture_frame_rate=60
        )

        parameters_channel = EnvironmentParametersChannel()
        parameters_channel.set_float_parameter("angle", self.angle)
        self.env = UnityEnvironment(
            file_name=self.file_name,
            side_channels=[config_channel, parameters_channel],
            # no_graphics=True,
        )
        # Start interacting with the environment.
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def reset(self):
        self.env.reset()
        self.num_died = 0
        self.agent_id_to_index = None
        obs, _, _, _ = self.observe()
        return obs

    def step(self, action):
        if self.last_idx:
            self.env.set_actions(self.behavior_name, action[self.last_idx, :])
        self.env.step()
        return self.observe()

    def observe(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        self.last_idx = None
        if decision_steps:
            if not self.agent_id_to_index:
                self.agent_id_to_index = decision_steps.agent_id_to_index
            idx = [self.agent_id_to_index[a] for a in decision_steps.agent_id]
            self.states[idx, 0] = 1 - decision_steps.obs[0][:, 1]  # left
            self.states[idx, 1] = 1 - decision_steps.obs[1][:, 1]  # forward
            self.states[idx, 2] = 1 - decision_steps.obs[2][:, 1]  # right
            self.last_idx = idx

        reward = sum(decision_steps.reward) + sum(terminal_steps.reward)

        if terminal_steps:
            self.num_died += len(terminal_steps)
        done = self.num_died == self.num_agents

        info = {}
        return self.states, reward, done, info
