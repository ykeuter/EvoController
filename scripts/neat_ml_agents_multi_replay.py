import os
import time
import click
import neat
import dotenv
import pickle

from neat.nn import FeedForwardNetwork

from evo_controller.worlds.ml_agents_multi_world \
    import MlAgentsMultiWorld


def run(config_file, checkpoint_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    fn = None
    # fn = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\searchlight"
    world = MlAgentsMultiWorld(file_name=fn, training=False)
    world.connect()
    # pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    # phenos = [FeedForwardNetwork.create(g, config)
    #           for g in pop.population.values()]
    with open(checkpoint_file, "rb") as f:
        g = pickle.load(f)
        phenos = [FeedForwardNetwork.create(g, config)]
    tic = time.perf_counter()
    for _ in range(10):
        fitnesses = world.evaluate(phenos)
        print(fitnesses)
    # print(sum(fitnesses) / len(fitnesses))
    toc = time.perf_counter()
    print("Replay took {} seconds.".format(toc - tic))
    world.disconnect()


if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__),
                                "../results/20210825181816")
    config_file = os.path.join(results_path, "neat-ml-agents-multi.cfg")
    check_file = os.path.join(results_path, "neat-ml-agents-multi-689")
    check_file = os.path.join(results_path, "winner.pickle")
    run(config_file, check_file)
