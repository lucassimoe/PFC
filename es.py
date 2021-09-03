import pickle
import datetime
import random
import argparse
import numpy
import wandb
from deap_custom import toolbox, tools, algorithms


parser = argparse.ArgumentParser(description="Evolution Strategies. ")
parser.add_argument("--env", default="Humanoid-v2")
parser.add_argument("--render", type=bool, default=False)

args = parser.parse_args()


wandb.init(project="Evolution-Estrategy", entity="lucas-simoes")
config = wandb.config
config.LAMBDA = 16
config.MU = 2
config.num_gen = 100
config.mutate_pop = 0.6
config.cross_prob = 0.3


def main():
    random.seed(42)
    MU, LAMBDA = config.MU, config.LAMBDA  # aumentar a população
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=config.cross_prob,
        mutpb=config.mutate_pop,
        ngen=config.num_gen,
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
