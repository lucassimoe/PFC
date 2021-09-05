from evostra.algorithms.evolution_strategy import EvolutionStrategy
import multiprocessing as mp
from tqdm import tqdm


class EvolutionEstrategyCustom(EvolutionStrategy):
    def run(self, iterations, print_step=10, logger=None):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        pbar = tqdm(range(iterations))

        for iteration in pbar:

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)
            stats = {
                "max": max(rewards),
                "avg": sum(rewards) / len(rewards),
                "min": min(rewards),
            }
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
