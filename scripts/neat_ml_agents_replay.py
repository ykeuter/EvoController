# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os
import time
import click
import neat
import dotenv

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet

# https://github.com/microsoft/vscode-python/issues/14570
from evo_controller.ml_agents_world \
    import MlAgentsWorld  # pylint: disable=import-error

dotenv.load_dotenv()


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs


def run(config_file, checkpoint_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # world = MlAgentsWorld(os.getenv('UNITY_ENV_EXE_DIR'))
    fn = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\WF3"
    # fn = None
    # angles = [-45, -30, 0, 30, 45]
    angles = [0]
    envs = [
        MlAgentsWorld(fn, worker_id=i, training=False, num_inputs=3, angle=a)
        for i, a in enumerate(angles)
    ]
    for w in envs:
        w.connect()
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=envs, batch_size=len(envs)
    )

    pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    s = 0
    tic = time.perf_counter()
    for genome in pop.population.values():
        f = evaluator.eval_genome(genome, config)
        s += f
        print(f)
    toc = time.perf_counter()
    print(s / len(pop.population))
    print("Replay took {} seconds.".format(toc - tic))

    for w in envs:
        w.disconnect()


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__),
                                "../results/20210114153314")
    config_file = os.path.join(results_path, "neat_ml_agents.cfg")
    check_file = os.path.join(results_path, "neat_ml_agents-109")
    run(config_file, check_file)
