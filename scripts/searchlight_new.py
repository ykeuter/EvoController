import time
from pathlib import Path
from datetime import datetime
import statistics

from neuro_evo_devo.population import Population
from neuro_evo_devo.phenotype import Phenotype
from neuro_evo_devo.neural_net import NeuralNet
from evo_controller.worlds.ml_agents_multi_world \
    import MlAgentsMultiWorld


FN = None
FN = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\searchlight"
LOG_DIR = Path("C:/Users/ykeuter/Projects/NeuroEvoDevo/logs") / \
    datetime.now().strftime("%Y%m%d%H%M")


def run():
    world = MlAgentsMultiWorld(file_name=FN, time_scale=20)
    world.connect()

    global generation
    generation = 0
    LOG_DIR.mkdir()

    def eval_genomes(genomes):
        global generation
        input_coords = [
            (-.75, .75), (-.25, .75), (.25, .75), (.75, .75),
            (-.75, .25), (-.25, .25), (.25, .25), (.75, .25),
            (-.75, -.25), (-.25, -.25), (.25, -.25), (.75, -.25),
            (-.75, -.75), (-.25, -.75), (.25, -.75), (.75, -.75),
        ]
        output_coords = [(0, 1), (1, 0)]
        phenos = [Phenotype(input_coords, output_coords, g) for g in genomes]
        brains = [NeuralNet(p) for p in phenos]
        fitnesses = world.evaluate(brains)
        len_genomes = [len(g.genes) for g in genomes]
        len_nn = [len(b.eval_order) - len(output_coords) for b in brains]
        print("gen: {} | "
              "fitness: ({}, {}, {}) | "
              "gene_size: ({}, {}, {}) | "
              "nn_size: ({}, {}, {})".format(
                  generation,
                  min(fitnesses), max(fitnesses), statistics.mean(fitnesses),
                  min(len_genomes), max(len_genomes),
                  statistics.mean(len_genomes),
                  min(len_nn), max(len_nn), statistics.mean(len_nn)
              ))
        generation += 1
        return fitnesses

    pop = Population(64, .3)

    tic = time.perf_counter()
    pop.run(eval_genomes, 1000)
    toc = time.perf_counter()
    print("Evolution took {} seconds.".format(toc - tic))
    world.disconnect()


if __name__ == "__main__":
    run()
