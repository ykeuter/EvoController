from neat import DefaultGenome
from configparser import ConfigParser
from .base_population import BasePopulation


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

    def activate(obs):
        pass
