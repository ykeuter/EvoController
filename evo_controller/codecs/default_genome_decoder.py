import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene


class DefaultGenomeDecoder():
    def as_default_genome(self, dct):
        if "activation" in dct:
            n = DefaultNodeGene(dct["key"])
            for a in n._gene_attributes:
                setattr(n, a.name, dct[a.name])
            return n
        if "from" in dct:
            c = DefaultConnectionGene((dct["from"], dct["to"]))
            for a in c._gene_attributes:
                setattr(c, a.name, dct[a.name])
            return c
        if "fitness" in dct:
            g = neat.DefaultGenome(dct["key"])
            g.fitness = dct["fitness"]
            g.nodes = {n.key: n for n in dct["nodes"]}
            g.connections = {c.key: c for c in dct["connections"]}
            return g
        return dct
