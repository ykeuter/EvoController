from evo_controller.codecs.default_genome_encoder import (
    DefaultGenomeEncoder
)
import pickle

folder = \
    "C:\\Users\\ykeuter\\Projects\\EvoController\\results\\20210826111846\\"
with open(folder + "winner.pickle", "rb") as f:
    g = pickle.load(f)
print(DefaultGenomeEncoder().encode(g))
