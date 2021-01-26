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

import multiprocessing
import os
import shutil
import datetime
import time
import click
import neat
import dotenv
import pathlib

# import torch
import numpy as np

from pytorch_neat.activations import tanh_activation
from pytorch_neat.adaptive_net import AdaptiveNet
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter

# https://github.com/microsoft/vscode-python/issues/14570
from evo_controller.ml_agents_world \
    import MlAgentsWorld  # pylint: disable=import-error

dotenv.load_dotenv()


def make_net(genome, config, _batch_size):
    # forward, right, back, left
    # input_coords = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
    # hidden_coords = [[0.0, .5], [.5, 0.0], [0.0, -.5], [-.5, 0.0]]
    # output_coords = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
    # left, forward, right
    input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    hidden_coords = [[-2.0, 1.0], [-1, 1.0], [1.0, 1.0], [2.0, 1.0]]
    output_coords = [[0.0, 2.0]]
    return AdaptiveNet.create(
        genome,
        config,
        input_coords=input_coords,
        hidden_coords=hidden_coords,
        output_coords=output_coords,
        batch_size=5,
        device='cpu'
    )


def activate_net(net, states, debug=False, step_num=0):
    outputs = net.activate(states[0]).cpu().numpy()
    return [outputs]


def run(config_file, log_path, n_generations=1000, n_processes=1):
    shutil.copy2(config_file, log_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # world = MlAgentsWorld(os.getenv('UNITY_ENV_EXE_DIR'))
    fn = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\WF4"
    # fn = None
    # angles = [-45, -30, 0, 30, 45]
    angles = [0]
    envs = [
        MlAgentsWorld(fn, worker_id=i, training=True, num_inputs=3, angle=a,
                      num_agents=5)
        for i, a in enumerate(angles)
    ]
    for w in envs:
        w.connect()
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=envs, batch_size=len(envs)
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome,
                ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:
        def eval_genomes(genomes, config):
            for _, genome in genomes:
                genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # fn = log_path / "hyper2_ml_agents.log"
    # logger = LogReporter(fn, evaluator.eval_genome)
    # pop.add_reporter(logger)

    prefix = log_path / "hyper2_ml_agents-"
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
    config_file = root / "config/hyper2_ml_agents.cfg"
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = root / "results" / dt
    log_path.mkdir()
    run(config_file, log_path)
