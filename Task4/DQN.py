import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from DQN_algo_modified import DQN
from open_spiel.python.algorithms import random_agent
import numpy as np
from open_spiel.python.algorithms import random_agent
import time


# Create the environment
env = rl_environment.Environment("dots_and_boxes(num_rows=7,num_cols=7)")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_size = num_actions


agents = [
    DQN(player_id=idx, num_actions=num_actions, state_representation_size=info_state_size, hidden_layers_sizes=[250])
    for idx in range(num_players)
]

start = time.time()

for cur_episode in range(25000):
    if cur_episode % 1000 == 0:
        print(f"Episodes: {cur_episode}")
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = [int(x) for x in env.get_state.dbn_string()]
        agent_output = agents[player_id].step_mod(time_step, state)
        time_step = env.step([agent_output.action])
    # Episode is over, step all agents with final info state.
    for agent in agents:
        state = [int(x) for x in env.get_state.dbn_string()]
        agent.step_mod(time_step, state)

end = time.time()
trainingtime = end - start
print("Done!")
print("trainingtime = ", trainingtime)
agents[0].save("q_network7x7mod_0.pt", "opt7x7mod_0.pt")
agents[1].save("q_network7x7mod_1.pt", "opt7x7mod_1.pt")


eval_agents = [agents[0], random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]
num_winsDQN = 0
num_winsRandom = 0

for itt in range(200):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == 0:
            state = [int(x) for x in env.get_state.dbn_string()]
            agent_output = eval_agents[player_id].step_mod(time_step, state, is_evaluation=True)
        else:
            agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
        #if player_id == 0:
        #    print(agent_output.probs)
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







