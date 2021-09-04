from deap import creator, base, tools, algorithms
import math
import array
import multiprocessing
from env_def import layer_sizes, generateES, checkStrategy, fitness

IND_SIZE = math.prod(
    layer_sizes
)  # tamanho do individuo, sendo a quantidade de pesos (376*32*16*17)
MIN_VALUE = -1
MAX_VALUE = 1
MIN_STRATEGY = -0.5
MAX_STRATEGY = 0.5


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create(
    "Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None
)
creator.create("Strategy", array.array, typecode="d")

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

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.4)
# toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register(
    "mutate",
    tools.mutGaussian,
    mu=0,
    sigma=0.2,
    indpb=0.6,
)  # aumentar um pouco o ruido
toolbox.register("select", tools.selRoulette)  # trocar pelo metodo roleta
toolbox.register("evaluate", fitness)

pool = multiprocessing.Pool(processes=8)
toolbox.register("map", pool.map)

toolbox.decorate("mate", checkStrategy(MAX_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MAX_STRATEGY))
