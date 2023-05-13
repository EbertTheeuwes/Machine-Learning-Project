import pyspiel
import numpy as np
import sys

import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
import io
from open_spiel.python.algorithms import random_agent
from dotsandboxes_agentDQN import Agent

env = rl_environment.Environment("dots_and_boxes(num_rows=5,num_cols=5)")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_size = env.observation_spec()["info_state"][0]
print("number of actions ", num_actions)
print("info_state_size ", info_state_size)
dqn_agent = dqn.DQN(player_id=0, num_actions=num_actions, state_representation_size=info_state_size)
model = torch.load("/Users/maxsebrechts/Desktop/Master/MLproj/Machine-Learning-Project/Task4/q_network5x5_0.pt")
model.eval()
dqn_agent._q_network = model
eval_agents = [dqn_agent, random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]

num_winsDQN = 0
num_winsRandom = 0
for itt in range(1):
    time_step = env.reset()
    print(time_step)
    print(time_step.observations["info_state"])
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
        if player_id == 0:
            print(agent_output.probs)
        if time_step.rewards[0] == 1:
            num_winsDQN += 1
        if time_step.rewards[1] == 1:
            num_winsRandom += 1

print("wins DQN: ", num_winsDQN)
print("wins Random: ", num_winsRandom)


