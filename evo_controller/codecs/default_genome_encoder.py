import json
from neat import DefaultGenome


class DefaultGenomeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DefaultGenome):
            nodes = [
                self.gene_to_dict(g, {"key": k})
                for k, g in obj.nodes.items()
            ]
            conns = [
                self.gene_to_dict(g, {"from": a, "to": b})
                for (a, b), g in obj.connections.items()
            ]
            return {
                "key": obj.key,
                "fitness": obj.fitness,
                "nodes": nodes,
                "connections": conns
            }
        return super().default(obj)

    def gene_to_dict(cls, g, dct):
        for att in g._gene_attributes:
            dct[att.name] = getattr(g, att.name)
        return dct
