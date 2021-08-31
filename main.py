from evostra import EvolutionStrategy
from evostra.models import FeedForwardNetwork
import pickle

from numpy import datetime_as_string
from environment import make_get_reward, env_info
from evolution_strategy import EvolutionEstrategyCustom

import argparse
import wandb
import datetime
parser = argparse.ArgumentParser(description='Evolution Strategies. ')
parser.add_argument('--env',default="Humanoid-v2")
parser.add_argument('--render',type=bool,default=False)
parser.add_argument('--checkPoint',default=None)

args = parser.parse_args()

observationSpace, actionSpace = env_info(args.env)

wandb.init(project='Evolution-Estrategy', entity='lucas-simoes')
config = wandb.config
config.population_size = 30
config.sigma = 0.1
config.learning_rate = 0.001
config.decay = 0.995
config.num_threads = 8
config.layer_sizes = [observationSpace, 400, 300, actionSpace]
config.env = args.env
config.iterations = int(3e4)



# A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
model = FeedForwardNetwork(layer_sizes=config.layer_sizes)

if args.checkPoint is not None:    
    with open(args.checkPoint,'rb') as fp:
        model.set_weights(pickle.load(fp))
        print("loaded checkPoint: "+args.checkPoint)

get_reward = make_get_reward(config.env, model,args.render)
# if your task is computationally expensive, you can use num_threads > 1 to use multiple processes;
# if you set num_threads=-1, it will use number of cores available on the machine; Here we use 1 process as the
#  task is not computationally expensive and using more processes would decrease the performance due to the IPC overhead.
es = EvolutionEstrategyCustom(
    model.get_weights(),
    get_reward,
    population_size=config.population_size,
    sigma=config.sigma,
    learning_rate=config.learning_rate,
    decay=config.decay,
    num_threads=config.num_threads,
)
es.run(config.iterations, print_step=100, logger=wandb)

model_path = "models/"+str(datetime.datetime.now()).replace(" ","_").replace(":","-").split(".")[0]+"_"+config.env +".pkl"
with open(model_path, 'wb') as fp:
    pickle.dump(es.get_weights(), fp)
    print("Saved "+model_path)
#while True:
#   print(get_reward(es.get_weights(),True))