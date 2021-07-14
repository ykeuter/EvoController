import pathlib
import logging.config
import json
import dotenv
import os

from evo_controller.worlds.open_world import OpenWorld
from evo_controller.population.alpha import Alpha

dotenv.load_dotenv()


def replay(config_file, gene_file):
    p = Alpha()
    p.config(config_file)
    p.load_gene_pool(gene_file)
    w = OpenWorld(p)
    w.connect()
    w.run()


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/alpha.cfg"
    gene_file = root / "genomes.txt"
    with open(root / "log_config.json") as fp:
        logging.config.dictConfig(json.load(fp))
    replay(config_file, gene_file)
