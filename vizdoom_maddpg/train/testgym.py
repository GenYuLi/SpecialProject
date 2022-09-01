import gym
from vizdoom import gym_wrapper
from gym import wrappers
import time
from os import system
if __name__  == '__main__':
    env = gym.make("VizdoomBasic-v0",host=2)
    #env = gym.make("VizdoomDeathmatch-v0",host=2)
    # Rendering random rollouts for ten episodes
    act = env.action_space.sample()
    
    #print(act["continuous"])
    #print(act["binary"])
    #print()
    #system("pause")
    for _ in range(100):
        done_n = False
        obs = env.reset()
        for _ in  range(5000):
            act = env.action_space.sample()
            print(act)
            obs, rew, done, info = env.step(act)
            done_n = all(done)
            if(done_n):
                break
            time.sleep(0.1)
            env.render()