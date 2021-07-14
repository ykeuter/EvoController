from evo_controller.codecs.default_genome_decoder import DefaultGenomeDecoder
import json


decoder = DefaultGenomeDecoder()
with open("genomes.txt") as f:
    for line in f:
        print("line:")
        d = json.loads(line)
        g = json.loads(line, object_hook=decoder.as_default_genome)
