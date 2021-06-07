from neat import DefaultGenome
from configparser import ConfigParser
from .base_population import BasePopulation
from .agent import Agent


class Alpha(BasePopulation):
    def __init__(self):
        self.agents = {}
        self.genome_type = DefaultGenome
        self.genome_config = None

    def config(self, fn):
        parameters = ConfigParser()
        parameters.read_file(fn)
        genome_dict = dict(parameters.items(self.genome_type.__name__))
        self.genome_config = self.genome_type.parse_config(genome_dict)

    def activate(self, decision_steps):
        print("activate")
        tup = ActionTuple(continuous=action[self.last_idx, :])

    def terminate(self, terminal_steps):
        print("terminate")
        for id in terminal_steps.agent_id:
            del self.agents[id]

    def add_agent(self, id, parent1_id, parent2_id):
        # parent2 is not supported
        genome = self.genome_type(id)
        if parent1_id == 0:
            genome.configure_new(self.genome_config)
        else:
            parent = self.agents[parent1_id].genotype
            genome.configure_crossover(parent, parent, self.genome_config)
            genome.mutate(self.genome_config)
        # phenotype
        self.agents[id] = Agent(genome, phenotype)
