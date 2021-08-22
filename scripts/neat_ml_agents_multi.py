import shutil
import datetime
import time
import neat
import pathlib

from neat.nn import FeedForwardNetwork

from evo_controller.worlds.ml_agents_multi_world \
    import MlAgentsMultiWorld


def run(config_file, log_path, n_generations=1000, max_env_steps=None):
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
    world = MlAgentsMultiWorld(config.pop_size, fn, training=True)
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

    prefix = log_path / "neat-ml-agents-multi-"
    checker = neat.Checkpointer(10, None, filename_prefix=prefix)
    pop.add_reporter(checker)

    tic = time.perf_counter()
    pop.run(eval_genomes, n_generations)
    toc = time.perf_counter()
    print("Evolution took {} seconds.".format(toc - tic))
    world.disconnect()


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent
    config_file = root / "config/neat-ml-agents-multi.cfg"
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = root / "results" / dt
    log_path.mkdir()
    run(config_file, log_path)