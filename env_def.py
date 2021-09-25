import gym
from gym.spaces.discrete import Discrete
import random
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


max_step = 100000
observationSpace, actionSpace = env_info("Humanoid-v2")
# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
layer_sizes = [observationSpace, 64, 64, actionSpace]
model = FeedForwardNetwork(layer_sizes=layer_sizes)


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
    env = gym.make("Humanoid-v2")  # criando ambiente para cada indivíduo "pesos"
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
            # print(info)
            # env.reset()
            # reward = 0
            break

    env.close()
    return (reward,)


def simulate_test(individual, render):
    env = gym.make("Humanoid-v2")  # criando ambiente para cada indivíduo "pesos"
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
            print(reward)
            env.reset()
            reward = 0
            # break

    # env.close()
    return (reward,)
