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

import click
import gym
import neat

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet

# from airsim_controller.airsim_world import AirsimWorld
from evo_controller.ml_agents_world import MlAgentsWorld


def make_env():
    return gym.make("CartPole-v0")


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs


@click.command()
@click.option("--n_generations", type=int, default=100)
@click.option("--max_env_steps", type=int, default=None)
def run(n_generations, max_env_steps):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__),
        "../config/neat_ml_agents.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    world = MlAgentsWorld()
    world.connect()
    evaluator = MultiEnvEvaluator(
        make_net, activate_net, max_env_steps=max_env_steps,
        envs=[world]
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    log_path = os.path.join(os.path.dirname(__file__),
        "../results/neat_ml_agents.log")
    logger = LogReporter(log_path, evaluator.eval_genome)
    pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)
    world.disconnect()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
