from neat import DefaultGenome
from configparser import ConfigParser
from .base_population import BasePopulation
from .agent import Agent
from neat.nn import FeedForwardNetwork
import numpy as np
import logging
import json
from mlagents_envs.base_env import ActionTuple
from evo_controller.codecs.default_genome_encoder import DefaultGenomeEncoder
from evo_controller.codecs.default_genome_decoder import DefaultGenomeDecoder


class Alpha(BasePopulation):
    def __init__(self):
        self.agents = {}
        self.genome_config = None
        self.logger = logging.getLogger(__name__)
        self.encoder = DefaultGenomeEncoder(separators=(",", ":"))

    def config(self, fn):
        parameters = ConfigParser()
        with open(fn) as f:
            parameters.read_file(f)
        genome_dict = dict(parameters.items(DefaultGenome.__name__))
        self.genome_config = DefaultGenome.parse_config(genome_dict)

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
            self.logger.info("death|{}".format(id))
            # print("terminate {}".format(id))
            del self.agents[id]

    def add_agent(self, id, parent1_id, parent2_id):
        # parent2 is not supported
        # print("conceive {} ({})".format(id, parent1_id))
        genome = DefaultGenome(id)
        genome.fitness = -1
        if parent1_id < 0:
            genome.configure_new(self.genome_config)
        else:
            parent = self.agents[parent1_id].genotype
            genome.configure_crossover(parent, parent, self.genome_config)
            genome.mutate(self.genome_config)
        self.logger.info("birth|{}|{}|{}|{}".format(
            id, parent1_id, parent2_id, self.encoder.encode(genome)))
        pheno = FeedForwardNetwork.create(genome, self)
        self.agents[id] = Agent(genome, pheno)

    def set_gene_pool(self, genes):
        # parent2 is not supported
        # print("conceive {} ({})".format(id, parent1_id))
        self.agents = {
            i: Agent(g, FeedForwardNetwork.create(g, self))
            for i, g in enumerate(genes)
        }

    def load_gene_pool(self, fn):
        decoder = DefaultGenomeDecoder()
        with open(fn) as f:
            genes = [
                json.loads(line, object_hook=decoder.as_default_genome)
                for line in f
            ]
        self.set_gene_pool(genes)

    def __len__(self):
        return len(self.agents)
