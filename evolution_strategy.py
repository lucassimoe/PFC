from evostra.algorithms.evolution_strategy import EvolutionStrategy
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pickle
import wandb


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)


class EvolutionEstrategyCustom(EvolutionStrategy):
    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = (
                (self.get_reward, self._get_weights_try(self.weights, p))
                for p in population
            )
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))

        infos = np.array(rewards)
        rewards = infos[:, 0]
        steps = infos[:, 1]
        return rewards, steps

    def _update_weights(self, rewards, population, k):
        rewards = np.array(rewards[:k])
        population = np.array(population[:k])
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (k * self.SIGMA)
            self.weights[index] = (
                w + update_factor * np.dot(layer_population.T, rewards).T
            )
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=10, logger=None, k=None):
        if k is None:
            k = self.POPULATION_SIZE
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        pbar = tqdm(range(iterations))
        rewards_aux = 0
        total_steps = 0
        cgen = 0
        while total_steps < iterations * 1000000:
            cgen += 1
            population = self._get_population()
            rewards, steps = self._get_rewards(pool, population)
            # print(rewards)
            # print("aaaa")
            total_steps += np.sum(steps)
            # print(total_steps)
            # ordene os índices em vez dos elementos em si
            indices = list(range(len(rewards)))
            indices.sort(
                key=lambda i: rewards[i], reverse=True
            )  # ordene os índices com relação ao seu respectivo valor em x
            rewards = [rewards[i] for i in indices]
            population = [population[i] for i in indices]
            self._update_weights(rewards, population, k)
            reward_base, step_base = self.get_reward(self.weights)
            if reward_base > rewards_aux:
                with open("best_ind.pkl", "wb") as fp:
                    pickle.dump(self.weights, fp)
                wandb.save("best_ind.pkl")
                rewards_aux = reward_base

            stats = {
                "max": max(rewards[:k]),
                "avg": sum(rewards[:k]) / len(rewards[:k]),
                "min": min(rewards[:k]),
                "base": reward_base,
            }
            pbar.set_postfix(stats)
            logger.log(stats)
            # if (iteration + 1) % print_step == 0:
            #     print(
            #         "iter %d. reward: %f"
            #         % (iteration + 1, self.get_reward(self.weights))
            #     )
        print(cgen)
        if pool is not None:
            pool.close()
            pool.join()
