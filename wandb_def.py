import wandb

wandb.init(project="Evolution-Estrategy", entity="lucas-simoes")
config = wandb.config
config.LAMBDA = 20
config.MU = 2
config.sigma = 0.3
config.mutate_atribute = 0.2
config.num_threads = 8
config.num_gen = 200
config.cross_prob = 0.2
config.cross_mate = 0.1
config.mutate_pop = 0.4
config.max_step = 100000
