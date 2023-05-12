import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import random_agent
import numpy as np
from open_spiel.python.algorithms import random_agent


# Create the environment
env = rl_environment.Environment("dots_and_boxes(num_rows=5,num_cols=5)")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_size = env.observation_spec()["info_state"][0]


agents = [
    dqn.DQN(player_id=idx, num_actions=num_actions, state_representation_size=info_state_size)
    for idx in range(num_players)
]

for cur_episode in range(25000):
    if cur_episode % 1000 == 0:
        print(f"Episodes: {cur_episode}")
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        time_step = env.step([agent_output.action])
    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)
print("Done!")

eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]
num_winsDQN = 0
num_winsRandom = 0

for itt in range(200):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        # Note the evaluation flag. A Q-learner will set epsilon=0 here.
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])

    if time_step.rewards[0] == 1:
        num_winsDQN += 1
    if time_step.rewards[1] == 1:
        num_winsRandom += 1

print("wins DQN: ", num_winsDQN)
print("wins Random: ", num_winsRandom)

"""
for cur_episode in range(25000):
    if cur_episode % 1000 == 0:
        print(f"Episodes: {cur_episode}")
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations
"""







