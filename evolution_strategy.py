from evostra.algorithms.evolution_strategy import EvolutionStrategy
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


class EvolutionEstrategyCustom(EvolutionStrategy):
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

        for iteration in pbar:

            population = self._get_population()
            rewards = self._get_rewards(pool, population)
            # print(rewards)
            # print(population)
            # ordene os índices em vez dos elementos em si
            indices = list(range(len(rewards)))
            indices.sort(
                key=lambda i: -rewards[i]
            )  # ordene os índices com relação ao seu respectivo valor em x
            rewards = [rewards[i] for i in indices]
            population = [population[i] for i in indices]
            # print(order_rewards)
            # print(order_population)
            self._update_weights(rewards, population, k)
            stats = {
                "max": max(rewards),
                "avg": sum(rewards) / len(rewards),
                "min": min(rewards),
            }
            if iteration == iterations / 4:
                print(self.learning_rate)
            if iteration == iterations / 2:
                print(self.learning_rate)
            pbar.set_postfix(stats)
            logger.log(stats)
            # if (iteration + 1) % print_step == 0:
            #     print(
            #         "iter %d. reward: %f"
            #         % (iteration + 1, self.get_reward(self.weights))
            #     )
        if pool is not None:
            pool.close()
            pool.join()
