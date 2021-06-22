import logging
import logging.config
import json
from neat import DefaultGenome
from configparser import ConfigParser
from evo_controller.codecs.default_genome_encoder import DefaultGenomeEncoder


with open("log_config.json") as fp:
    logging.config.dictConfig(json.load(fp))

logging.getLogger("evo_controller").info(123)

parameters = ConfigParser()
with open("config/alpha.cfg") as f:
    parameters.read_file(f)
genome_dict = dict(parameters.items(DefaultGenome.__name__))
genome_config = DefaultGenome.parse_config(genome_dict)

genome = DefaultGenome(777)
genome.fitness = -1
genome.configure_new(genome_config)

print(genome.__dict__)
logging.getLogger("evo_controller").info(
    "{}".format(DefaultGenomeEncoder(separators=(",", ":")).encode(genome))
)
