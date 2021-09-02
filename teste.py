from evostra.models import FeedForwardNetwork
import pickle

from env_def import simulate, env_info
import argparse
import gym
import deap_custom

parser = argparse.ArgumentParser(description="Evolution Strategies.")
parser.add_argument("--env", default="Humanoid-v2")
parser.add_argument("--model", default=None)

args = parser.parse_args()

observationSpace, actionSpace = env_info(args.env)


# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=[observationSpace, 64, 64, actionSpace])


with open(args.model + ".pkl", "rb") as fb:
    weights = pickle.load(fb)

ind = weights["halloffame"][0]

model.set_weights(ind)
get_reward = simulate(ind, True)
# print(get_reward)
# while True:
#     get_reward = simulate(ind, True)
# with open(args.model,'rb') as fp:
#   model.set_weights(pickle.load(fp))

# env = gym.make(args.env)
# obs = env.reset()
# reward = 0

# while True:
#     env.render()
#     # print(model.predict(obs))
#     action = model.predict(obs)  # your agent here (this takes random actions)
#     obs, rew, done, info = env.step(action)
#     reward += rew

#     if done:
#         # print(step)
#         print(reward)
#         env.reset()
#         reward = 0
