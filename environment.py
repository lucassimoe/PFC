import numpy as np
import gym
from gym.spaces.discrete import Discrete
import glfw

def env_info(env_name):
    env = gym.make(env_name)

    observation_space = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        action_space = env.action_space.n
    else:
        action_space = env.action_space.shape[0]
    env.close()
    return observation_space, action_space
get_reward = None
def make_get_reward(env_name, model, _render):
    global   get_reward
    def get_reward(weights, render=_render):
        env = gym.make(env_name)
        #global model
        model.set_weights(weights)
        # here our best reward is zero
        reward = 0
        obs = env.reset()
        for step in range(100000):
            if render:
                env.render()
            #print(model.predict(obs))
            action = model.predict(obs) # your agent here (this takes random actions)

            obs, rew, done, info = env.step(action)
            reward+=rew

            if done:
                #print(step)
                break
        env.close()
        return reward
    return get_reward