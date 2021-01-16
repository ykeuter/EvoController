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
import shutil
import datetime
import time
import click
import neat
import dotenv
import pathlib
import numpy as np

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet

# https://github.com/microsoft/vscode-python/issues/14570
from evo_controller.ml_agents_world \
    import MlAgentsWorld  # pylint: disable=import-error

dotenv.load_dotenv()


def make_net(genome, config, bs):
    # return RecurrentNet.create(genome, config, bs)
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    # return np.array([[0, 0], [0, 0]])
    outputs = net.activate(states).numpy()
    return outputs


def run(config_file, log_path, n_generations=1000, max_env_steps=None):
    shutil.copy2(config_file, log_path)
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
    angles = [-45, -30, 0, 30, 45]
    # angles = [0]
    envs = [
        MlAgentsWorld(fn, worker_id=i, training=True, num_inputs=3, angle=a)
        for i, a in enumerate(angles)
    ]
    for w in envs:
        w.connect()
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, max_env_steps=max_env_steps, envs=envs,
        batch_size=len(envs)
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # fn = log_path / "neat_ml_agents.log"
    # logger = LogReporter(fn, evaluator.eval_genome)
    # pop.add_reporter(logger)

    prefix = log_path / "neat_ml_agents-"
    checker = neat.Checkpointer(10, None, filename_prefix=prefix)
    pop.add_reporter(checker)

    tic = time.perf_counter()
    pop.run(eval_genomes, n_generations)
    toc = time.perf_counter()
    print("Evolution took {} seconds.".format(toc - tic))
    for w in envs:
        w.disconnect()


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/neat_ml_agents.cfg"
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = root / "results" / dt
    log_path.mkdir()
    run(config_file, log_path)
