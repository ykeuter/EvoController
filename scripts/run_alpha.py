import pathlib
import logging.config
import json
import dotenv
import os

from evo_controller.worlds.open_world import OpenWorld
from evo_controller.population.alpha import Alpha

dotenv.load_dotenv()


def run(config_file):
    p = Alpha()
    p.config(config_file)
    fn = os.getenv('UNITY_ENV_EXE_DIR')
    w = OpenWorld(p, fn, True)
    w.connect()
    w.run()


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/alpha.cfg"
    with open(root / "log_config.json") as fp:
        logging.config.dictConfig(json.load(fp))
    run(config_file)
