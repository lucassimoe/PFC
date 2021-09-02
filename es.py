import pickle
import datetime
import random
import argparse
import numpy

from deap_custom import toolbox, tools, algorithms


parser = argparse.ArgumentParser(description="Evolution Strategies. ")
parser.add_argument("--env", default="Humanoid-v2")
parser.add_argument("--render", type=bool, default=False)

args = parser.parse_args()


# wandb.init(project="Evolution-Estrategy", entity="lucas-simoes")
# config = wandb.config
# config.LAMBDA = 40
# config.MU = 1
# config.sigma = 0.3
# config.mutateAtribute = 0.1
# config.numThreads = 8
# config.numGen = 200
# config.mutatePop = 0.9
# config.layer_sizes = [observationSpace, 64, 64, actionSpace]
# config.env = args.env


def main():
    random.seed(42)
    MU, LAMBDA = 2, 20  # aumentar a população
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuCommaLambda(
        pop,
        toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=0.2,
        mutpb=0.4,
        ngen=200,
        stats=stats,
        halloffame=hof,
    )

    modelPath = (
        str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
        + "_"
        + args.env
        + ".pkl"
    )
    with open(modelPath, "wb") as fp:
        sol = dict(population=pop, logbook=logbook, halloffame=hof)
        pickle.dump(sol, fp)
        print("Saved " + modelPath)
    return pop, logbook, hof


if __name__ == "__main__":
    main()
