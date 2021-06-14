from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np
from evo_controller.channels.birth_channel import BirthChannel


class OpenWorld:
    def __init__(self, population, file_name=None, training=True, worker_id=0):
        self.env = None
        self.behavior_name = None
        self.file_name = file_name
        self.training = training
        self.worker_id = worker_id
        self.population = population

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

        birth_channel = BirthChannel(self.population)

        self.env = UnityEnvironment(
            file_name=self.file_name,
            side_channels=[config_channel, birth_channel],
            worker_id=self.worker_id,
            # no_graphics=True,
        )
        # Start interacting with the environment.
        print("reset")
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def run(self):
        while True:
            decision_steps, terminal_steps = self.env.get_steps(
                self.behavior_name)
            if terminal_steps:
                self.population.terminate(terminal_steps)
            if decision_steps:
                actions = self.population.activate(decision_steps)
                self.env.set_actions(self.behavior_name, actions)
            self.env.step()
