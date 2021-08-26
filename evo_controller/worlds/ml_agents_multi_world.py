from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel \
    import EnvironmentParametersChannel
from mlagents_envs import logging_util
import numpy as np


class MlAgentsMultiWorld:
    def __init__(self, num_cases=4,
                 file_name=None, time_scale=None, worker_id=0):
        self.num_cases = num_cases
        self.env = None
        self.parameters_channel = EnvironmentParametersChannel()
        self.behavior_name = None
        self.file_name = file_name
        self.time_scale = time_scale
        self.worker_id = worker_id

    def connect(self):
        logging_util.set_log_level(logging_util.INFO)

        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(
            width=1024,
            height=576,
            quality_level=0,
            time_scale=self.time_scale,
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
        rewards = [0] * len(brains)
        num_done = 0
        self.env.step()  # make sure action buffer is empty
        self.parameters_channel.set_float_parameter("case_id", case_id)
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        if len(terminal_steps) > 0:
            raise ValueError
        num_agents = len(decision_steps)
        while num_done < num_agents:
            for i in decision_steps.agent_id:
                if i >= len(brains):
                    continue
                obs = decision_steps[i].obs
                # rewards[i] += decision_steps[i].reward
                if abs(decision_steps[i].reward) > 0:
                    raise ValueError
                action = brains[i].activate(np.ravel(obs))
                self.env.set_action_for_agent(
                    self.behavior_name,
                    i,
                    ActionTuple(continuous=np.array([action]))
                )
            self.env.step()
            decision_steps, terminal_steps = \
                self.env.get_steps(self.behavior_name)
            for i in terminal_steps.agent_id:
                if i >= len(brains):
                    continue
                rewards[i] += terminal_steps[i].reward
            num_done += len(terminal_steps)
            if num_done > num_agents:
                raise ValueError
        return rewards
