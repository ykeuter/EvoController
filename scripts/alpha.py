import pathlib

from evo_controller.worlds.open_world import OpenWorld
from evo_controller.population.alpha import Alpha


def run(config_file):
    p = Alpha()
    p.config(config_file)
    w = OpenWorld(p, None, False)
    w.connect()
    w.run()


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/alpha.cfg"
    run(config_file)
