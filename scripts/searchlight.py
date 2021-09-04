import shutil
import datetime
import time
import neat
import pathlib
import pickle

from neat.nn import FeedForwardNetwork

from evo_controller.worlds.ml_agents_multi_world \
    import MlAgentsMultiWorld

from evo_controller.codecs.default_genome_encoder import (
    DefaultGenomeEncoder
)


def run(config_file, log_path, n_generations=1000):
    shutil.copy2(config_file, log_path)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    fn = None
    fn = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\searchlight"
    world = MlAgentsMultiWorld(file_name=fn, time_scale=20)
    world.connect()

    def eval_genomes(genomes, config):
        phenos = [FeedForwardNetwork.create(g, config) for _, g in genomes]
        fitnesses = world.evaluate(phenos)
        for (_, genome), f in zip(genomes, fitnesses):
            genome.fitness = f

    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # fn = log_path / "neat_ml_agents.log"
    # logger = LogReporter(fn, evaluator.eval_genome)
    # pop.add_reporter(logger)

    prefix = log_path / "searchlight-"
    checker = neat.Checkpointer(10, None, filename_prefix=prefix)
    pop.add_reporter(checker)

    tic = time.perf_counter()
    winner = pop.run(eval_genomes, n_generations)
    toc = time.perf_counter()
    print("Evolution took {} seconds.".format(toc - tic))
    print(DefaultGenomeEncoder().encode(winner))
    world.disconnect()

    with open(log_path / "winner.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/searchlight.cfg"
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = root / "results" / dt
    log_path.mkdir()
    run(config_file, log_path)
