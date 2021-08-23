from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsMultiWorld:
    def __init__(self, pop_size, file_name=None, training=False, worker_id=0):
        self.pop_size = pop_size
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
        n = len(brains)
        rewards = [0] * n
        if n != self.pop_size:
            print("Pop size {} vs expected {}.".format(n, self.pop_size))
        num_done = 0
        self.env.reset()
        while num_done < len(brains):
            decision_steps, terminal_steps = \
                self.env.get_steps(self.behavior_name)
            for i in decision_steps.agent_id:
                if i >= n:
                    continue
                obs = decision_steps[i].obs
                rewards[i] += decision_steps[i].reward
                action = brains[i].activate(np.ravel(obs))
                # action = [0, 1]
                self.env.set_action_for_agent(
                    self.behavior_name,
                    i,
                    ActionTuple(continuous=np.array([action]))
                )
            for i in terminal_steps.agent_id:
                if i >= n:
                    continue
                # print(terminal_steps[i].reward)
                rewards[i] += terminal_steps[i].reward
            num_done += len(terminal_steps)
            self.env.step()
        # if n > self.pop_size:
        #     rewards[self.pop_size:] = [rewards[0]] * (n - self.pop_size)
        return rewards
