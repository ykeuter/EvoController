from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsMultiWorld:
    def __init__(self, file_name=None, training=True,
                 num_agents=1, num_inputs=1, worker_id=0):
        self.env = None
        self.behavior_name = None
        self.file_name = file_name
        self.training = training
        self.worker_id = worker_id

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(
            # width=84 if self.training else 512,
            # height=84 if self.training else 512,
            width=1024,
            height=576,
            quality_level=0,
            time_scale=100 if self.training else 1,
            target_frame_rate=-1,
            # capture_frame_rate=60
        )

        parameters_channel = EnvironmentParametersChannel()
        # parameters_channel.set_float_parameter("angle", self.angle)
        self.env = UnityEnvironment(
            file_name=self.file_name,
            side_channels=[config_channel, parameters_channel],
            worker_id=self.worker_id,
            # no_graphics=True,
        )
        # Start interacting with the environment.
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def evaluate(self, brains):
        rewards = [0] * len(brains)
        num_died = 0
        self.env.reset()
        while num_died < len(brains):
            decision_steps, terminal_steps = \
                self.env.get_steps(self.behavior_name)
            for ds in decision_steps:
                action = brains[ds.agent_id].activate(ds.obs)
                self.env.set_action_for_agent(
                    self.behavior_name,
                    ds.agent_id,
                    ActionTuple(continuous=action)
                )
            for ts in terminal_steps:
                rewards[ts.agent_id] = ts.reward
            num_died += len(terminal_steps)
            self.env.step()
        return rewards
