import sys
import numpy as np
import pandas as pd
#from agents.policy_search import PolicySearch_Agent
from agents.agent import DDPG
from task import Task

num_episodes = 1000
target_pos = np.array([0., 0., 100.])
task = Task(target_pos=target_pos)
agent = DDPG(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})  total_reword = {:7.3f}".format(
                i_episode, agent.score, agent.best_score, agent.total_reward), end="")  # [debug]
            break
    sys.stdout.flush()




