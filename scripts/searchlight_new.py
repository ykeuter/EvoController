import time

from neuro_evo_devo.population import Population
from neuro_evo_devo.phenotype import Phenotype
from neuro_evo_devo.neural_net import NeuralNet
from evo_controller.worlds.ml_agents_multi_world \
    import MlAgentsMultiWorld


def run():
    fn = None
    fn = "C:\\Users\\ykeuter\\Projects\\EvoWorld\\app\\searchlight"
    world = MlAgentsMultiWorld(file_name=fn, time_scale=20)
    world.connect()

    def eval_genomes(genomes):
        input_coords = [
            (-.75, .75), (-.25, .75), (.25, .75), (.75, .75),
            (-.75, .25), (-.25, .25), (.25, .25), (.75, .25),
            (-.75, -.25), (-.25, -.25), (.25, -.25), (.75, -.25),
            (-.75, -.75), (-.25, -.75), (.25, -.75), (.75, -.75),
        ]
        output_coords = [(0, 1), (1, 0)]
        brains = [
            NeuralNet(Phenotype(input_coords, output_coords, g))
            for g in genomes
        ]
        return world.evaluate(brains)

    pop = Population(64, .3)

    tic = time.perf_counter()
    pop.run(eval_genomes, 1000)
    toc = time.perf_counter()
    print("Evolution took {} seconds.".format(toc - tic))
    world.disconnect()


if __name__ == "__main__":
    run()
