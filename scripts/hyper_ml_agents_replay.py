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
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter

# https://github.com/microsoft/vscode-python/issues/14570
from evo_controller.ml_agents_world \
    import MlAgentsWorld  # pylint: disable=import-error

dotenv.load_dotenv()


def make_net(genome, config, _batch_size):
    # forward, right, back, left
    input_coords = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
    output_coords = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        # weight_threshold=0.4,
        batch_size=_batch_size,
        activation=tanh_activation,
        output_activation=tanh_activation,
        device="cpu",
    )


def activate_net(net, states, debug=False, step_num=0):
    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)
        print("W init: ", net.input_to_output[0])
    outputs = net.activate(states).cpu().numpy()
    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])
        print("Delta W: ", net.delta_w[0])
        print("W: ", net.input_to_output[0])
    return outputs


def run(config_file, checkpoint_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    world = MlAgentsWorld(os.getenv('UNITY_ENV_EXE_DIR'), training=False)
    # world = MlAgentsWorld()
    world.connect()
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=[world]
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

    world.disconnect()


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__),
                                "../results/20201220170126")
    config_file = os.path.join(results_path, "hyper_ml_agents.cfg")
    check_file = os.path.join(results_path, "hyper_ml_agents-999")
    run(config_file, check_file)
