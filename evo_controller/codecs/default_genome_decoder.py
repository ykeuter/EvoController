import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene


class DefaultGenomeDecoder():
    def __init__(self, config):
        self.config = config


def as_default_genome(self, dct):
    g = neat.DefaultGenome(dct["key"])
    g.fitness = dct["fitness"]
    g.nodes = [
        self.create_node(d) for d in dct["nodes"]
    ]
    g.connections = [
        self.create_connection(d) for d in dct["connections"]
    ]


def create_node(self, dct):
    n = DefaultNodeGene(dct["key"])
    for a in n._gene_attributes:
        setattr(n, a.name, dct[a.name])
    return n


def create_connection(self, dct):
    c = DefaultConnectionGene(dct["key"])
    for a in c._gene_attributes:
        setattr(c, a.name, dct[a.name])
    return c
