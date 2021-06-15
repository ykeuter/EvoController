from neat import DefaultGenome
from configparser import ConfigParser
from .base_population import BasePopulation
from .agent import Agent
from neat.nn import FeedForwardNetwork, feed_forward
import numpy as np
from mlagents_envs.base_env import ActionTuple


class Alpha(BasePopulation):
    def __init__(self):
        self.agents = {}
        self.genome_type = DefaultGenome
        self.genome_config = None

    def config(self, fn):
        parameters = ConfigParser()
        with open(fn) as f:
            parameters.read_file(f)
        genome_dict = dict(parameters.items(self.genome_type.__name__))
        self.genome_config = self.genome_type.parse_config(genome_dict)

    def activate(self, decision_steps):
        actions = np.array([
            self.agents[id].phenotype.activate(np.ravel(row))
            # [0]
            for id, row
            in zip(decision_steps.agent_id, decision_steps.obs[0])
        ])
        return ActionTuple(continuous=actions)

    def terminate(self, terminal_steps):
        for id in terminal_steps.agent_id:
            print("terminate {}".format(id))
            del self.agents[id]

    def add_agent(self, id, parent1_id, parent2_id):
        # parent2 is not supported
        print("conceive {} ({})".format(id, parent1_id))
        genome = self.genome_type(id)
        genome.fitness = -1
        if parent1_id < 0:
            genome.configure_new(self.genome_config)
        else:
            parent = self.agents[parent1_id].genotype
            genome.configure_crossover(parent, parent, self.genome_config)
            genome.mutate(self.genome_config)
        pheno = FeedForwardNetwork.create(genome, self)
        self.agents[id] = Agent(genome, pheno)
