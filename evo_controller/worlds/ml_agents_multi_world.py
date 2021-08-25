from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsMultiWorld:
    def __init__(self, num_cases=4, pop_size=1,
                 file_name=None, training=False, worker_id=0):
        self.num_cases = num_cases
        self.pop_size = pop_size
        self.env = None
        self.parameters_channel = EnvironmentParametersChannel()
        self.behavior_name = None
        self.file_name = file_name
        self.training = training
        self.worker_id = worker_id

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(
            width=1024,
            height=576,
            quality_level=0,
            time_scale=100 if self.training else 1,
            target_frame_rate=-1,
            # capture_frame_rate=60
        )

        self.env = UnityEnvironment(
            file_name=self.file_name,
            side_channels=[config_channel, self.parameters_channel],
            worker_id=self.worker_id,
            # no_graphics=True,
        )
        # Start interacting with the environment.
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]

    def disconnect(self):
        self.env.close()

    def evaluate(self, brains):
        list_of_rewards = [
            self._evaluate_case(brains, c) for c in range(self.num_cases)]
        return [sum(r) for r in zip(*list_of_rewards)]

    def _evaluate_case(self, brains, case_id):
        n = len(brains)
        rewards = [0] * n
        num_done = 0
        self.parameters_channel.set_float_parameter("case_id", case_id)
        self.env.reset()
        steps = 0
        while True:
            decision_steps, terminal_steps = \
                self.env.get_steps(self.behavior_name)
            if steps == 0:
                if len(terminal_steps) > 0:
                    raise ValueError
                # self.ts = []
            # self.ts.append(terminal_steps)
            for i in terminal_steps.agent_id:
                if i >= n:
                    continue
                rewards[i] += terminal_steps[i].reward
            num_done += len(terminal_steps)
            if num_done == self.pop_size:
                break
            if num_done > self.pop_size:
                raise ValueError

            for i in decision_steps.agent_id:
                if i >= n:
                    continue
                obs = decision_steps[i].obs
                # rewards[i] += decision_steps[i].reward
                if abs(decision_steps[i].reward) > 0:
                    raise ValueError
                action = brains[i].activate(np.ravel(obs))
                # action = [0, 1]
                self.env.set_action_for_agent(
                    self.behavior_name,
                    i,
                    ActionTuple(continuous=np.array([action]))
                )
            self.env.step()
            steps += 1
        return rewards
