from evostra.models import FeedForwardNetwork
import pickle

from environment import make_get_reward, env_info
from evolution_strategy import EvolutionEstrategyCustom

import argparse
import wandb
import datetime

parser = argparse.ArgumentParser(description="Evolution Strategies. ")
parser.add_argument("--seed", default=0)
parser.add_argument("--render", type=bool, default=False)
parser.add_argument("--pop_size", default=None)
parser.add_argument("--pop_best", default=None)
parser.add_argument("--sigma", default=None)
parser.add_argument("--learning_rate", default=None)
parser.add_argument("--decay", default=None)
parser.add_argument("--num_threads", default=-1)
parser.add_argument("--env", default="Humanoid-v2")
parser.add_argument("--it", default=50)

args = parser.parse_args()

observationSpace, actionSpace = env_info(args.env)

wandb.init(project="ES-Envs-test", entity="lucas-simoes")
config = wandb.config
config.seed = int(args.seed)
config.population_size = int(args.pop_size)
config.population_bests = int(args.pop_best)
config.sigma = float(args.sigma)
config.learning_rate = float(args.learning_rate)
config.decay = float(args.decay)
config.num_threads = int(args.num_threads)
config.layer_sizes = [observationSpace, 50, actionSpace]
config.env = args.env
config.iterations = int(args.it)

# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=config.layer_sizes)

get_reward = make_get_reward(config.env, model, args.render)
# if your task is computationally expensive, you can use num_threads > 1 to use multiple processes;
# if you set num_threads=-1, it will use number of cores available on the machine; Here we use 1 process as the
#  task is not computationally expensive and using more processes would decrease the performance due to the IPC overhead.


es = EvolutionEstrategyCustom(
    model.get_weights(),
    get_reward,
    value_seed=config.seed,
    population_size=config.population_size,
    sigma=config.sigma,
    learning_rate=config.learning_rate,
    decay=config.decay,
    num_threads=config.num_threads,
)
es.run(config.iterations, logger=wandb, k=config.population_bests)

model_path = (
    "models/"
    + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
    + "_"
    + config.env
    + ".pkl"
)
with open(model_path, "wb") as fp:
    pickle.dump(es.get_weights(), fp)
    print("Saved " + model_path)

# Save a model file from the current directory
wandb.save(model_path)

# while True:
#   print(get_reward(es.get_weights(),True))
