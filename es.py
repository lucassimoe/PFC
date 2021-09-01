import numpy as np
import gym
from gym.spaces.discrete import Discrete

from evostra import EvolutionStrategy
from feed_forward_network import FeedForwardNetwork
import pickle

import wandb
import datetime

import math
import numpy
import array
import random
from deap import creator, base, tools, algorithms
import multiprocessing

import argparse

parser = argparse.ArgumentParser(description="Evolution Strategies. ")
parser.add_argument("--env", default="Humanoid-v2")
parser.add_argument("--render", type=bool, default=False)

args = parser.parse_args()


def env_info(env_name):
    env = gym.make(env_name)

    observation_space = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        action_space = env.action_space.n
    else:
        action_space = env.action_space.shape[0]
    env.close()
    return observation_space, action_space


observationSpace, actionSpace = env_info(args.env)

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
# config.iterations = int(3e4)

# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
layer_sizes = [observationSpace, 64, 64, actionSpace]
model = FeedForwardNetwork(layer_sizes=layer_sizes)


IND_SIZE = math.prod(
    layer_sizes
)  # tamanho do individuo, sendo a quantidade de pesos (376*32*16*17)
MIN_VALUE = -1
MAX_VALUE = 1
MIN_STRATEGY = -0.5
MAX_STRATEGY = 0.5
max_step = 100000

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create(
    "Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None
)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


toolbox = base.Toolbox()
toolbox.register(
    "individual",
    generateES,
    creator.Individual,
    creator.Strategy,
    IND_SIZE,
    MIN_VALUE,
    MAX_VALUE,
    MIN_STRATEGY,
    MAX_STRATEGY,
)


def fitness(individual):
    reward = simulate(individual, False)
    return reward


def selAverage(individuals, k, fit_attr="fitness"):
    ind = individuals[0]
    print("K = {}".format(k))
    fitness = abs(getattr(ind, fit_attr).values[0])
    for j in range(len(ind)):
        ind[j] = ind[j] * fitness
    sum_fitness = fitness

    for i in range(1, len(individuals)):
        fitness = abs(getattr(individuals[i], fit_attr).values[0])
        for j in range(len(individuals[i])):
            ind[j] += fitness * individuals[i][j]
        sum_fitness += fitness

    for j in range(len(ind)):
        ind[j] /= sum_fitness
    print("Mean Fitness: {}".format(sum_fitness / len(individuals)))
    return [ind for _ in range(k)]


def simulate(individual, render):
    env = gym.make(args.env)  # criando ambiente para cada indivíduo "pesos"
    model.set_weights(individual)  # alterando os pesos da rede
    # here our best reward is zero
    reward = 0
    obs = env.reset()
    for step in range(max_step):
        if render:
            env.render()
        obs, rew, done, info = env.step(model.predict(obs))
        reward += rew

        if done:
            break
    env.close()
    return (reward,)


toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
# toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register(
    "mutate",
    tools.mutGaussian,
    mu=0,
    sigma=0.3,
    indpb=0.1,
)  # aumentar um pouco o ruido
toolbox.register("select", selAverage)  # trocar pelo metodo roleta
toolbox.register("evaluate", fitness)

pool = multiprocessing.Pool(processes=8)
toolbox.register("map", pool.map)

toolbox.decorate("mate", checkStrategy(MAX_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MAX_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 1, 40  # aumentar a população
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
        cxpb=0,
        mutpb=0.9,
        ngen=1000,
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
