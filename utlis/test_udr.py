import gymnasium as gym
from env.custom_hopper import *

env = gym.make("CustomHopper-source-v0")
env.unwrapped.udr_enabled = True

for k in range(5):
    env.reset(seed=k)
    masses = env.unwrapped.model.body_mass
    print(k, "torso:", masses[1], "thigh/leg/foot:", masses[2], masses[3], masses[4])

env.close()
