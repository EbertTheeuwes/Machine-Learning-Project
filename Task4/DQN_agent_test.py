import pyspiel
import numpy as np
import sys

import torch
from open_spiel.python import rl_environment
import io
from open_spiel.python.algorithms import random_agent
from dotsandboxes_agentDQN import Agent
from dotsandboxes_agent_random import AgentRand

# test where other game is actually played (representation in DQN agent is 7x7)
# change dimensions here to play different game
game = pyspiel.load_game("dots_and_boxes(num_rows=7,num_cols=7)")
env = rl_environment.Environment("dots_and_boxes(num_rows=5,num_cols=5)")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_size = env.observation_spec()["info_state"][0]

dqn_agent = Agent(0)
rand_agent = AgentRand(1)
eval_agents = [dqn_agent, rand_agent]

num_winsDQN = 0
num_winsRandom = 0

for itt in range(200):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        #print("outer state")
        #print(env.get_state)

        action = eval_agents[player_id].step(env.get_state)

        for agent in eval_agents:
            if agent.player_id != player_id:
                agent.inform_action(env.get_state, player_id, action)
        #print("action in outer state ", action, "player id ", player_id)

        time_step = env.step([action])

    dqn_agent = Agent(0)
    rand_agent = AgentRand(1)
    eval_agents = [dqn_agent, rand_agent]

    if time_step.rewards[0] == 1:
        num_winsDQN += 1
    if time_step.rewards[1] == 1:
        num_winsRandom += 1

print("wins DQN: ", num_winsDQN)
print("wins Random: ", num_winsRandom)
print("win percentage: ", num_winsDQN/(num_winsRandom + num_winsDQN))
