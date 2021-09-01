from deap import creator, base, tools, algorithms
import gym
from gym.spaces.discrete import Discrete
import math
import random
import array
import multiprocessing
from feed_forward_network import FeedForwardNetwork


def env_info(env_name):
    env = gym.make(env_name)

    observation_space = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        action_space = env.action_space.n
    else:
        action_space = env.action_space.shape[0]
    env.close()
    return observation_space, action_space


envName = "Humanoid-v2"
observationSpace, actionSpace = env_info(envName)
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
    # print("K = {}".format(k))
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
    # print("Mean Fitness: {}".format(sum_fitness / len(individuals)))
    return [ind for _ in range(k)]


def simulate(individual, render):
    env = gym.make(envName)  # criando ambiente para cada indivíduo "pesos"
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
    indpb=0.2,
)  # aumentar um pouco o ruido
toolbox.register("select", selAverage)  # trocar pelo metodo roleta
toolbox.register("evaluate", fitness)

pool = multiprocessing.Pool(processes=8)
toolbox.register("map", pool.map)

toolbox.decorate("mate", checkStrategy(MAX_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MAX_STRATEGY))
