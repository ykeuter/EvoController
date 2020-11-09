from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs import logging_util
import numpy as np

logging_util.set_log_level(logging_util.INFO)

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(
    width=84,
    height=84,
    quality_level=5,
    time_scale=20,
    target_frame_rate=-1,
    capture_frame_rate=60)
env = UnityEnvironment(side_channels=[channel])
# Start interacting with the environment.
env.reset()
for _ in range(1000):
    for name, spec in env.behavior_specs.items():
        decision_steps, _ = env.get_steps(name)
        a = np.zeros((len(decision_steps), 2), dtype=np.float32)
        a[:, 0] = 1.0
        env.set_actions(name, a)
        env.step()
env.close()
