import pyspiel
import numpy as np
import sys

import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
import io
from open_spiel.python.algorithms import random_agent
from dotsandboxes_agentDQN import Agent

game = pyspiel.load_game("dots_and_boxes(num_rows=7,num_cols=7)")


# Create the environment
env = rl_environment.Environment("dots_and_boxes(num_rows=7,num_cols=7)")
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_size = env.observation_spec()["info_state"][0]
print("number of actions ", num_actions)
print("info_state_size ", info_state_size)
dqn_agent = dqn.DQN(player_id=0, num_actions=num_actions, state_representation_size=info_state_size, hidden_layers_sizes=200)
agent = Agent(0)


time_step = env.reset()
legal_actions5x5 = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 88, 89, 90, 91, 92, 93]
for action in legal_actions5x5:
    time_step = env.step([action])

print(env.get_state)
trans_actions = []
for action in range(5*6*2):
    trans_actions += [agent.action_transform(action, 5, 5)]

print(trans_actions)
print("is equal: ", trans_actions == legal_actions5x5)


"""

# Assuming 'model_path' is the path to your model file
with open("/Users/maxsebrechts/Desktop/Master/MLproj/Machine-Learning-Project/Task4/q_network.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())

model = torch.load("/Users/maxsebrechts/Desktop/Master/MLproj/Machine-Learning-Project/Task4/q_network.pt")
dqn_agent._q_network = model
eval_agents = [dqn_agent, random_agent.RandomAgent(1, num_actions, "Entropy Master 2000") ]
num_winsDQN = 0
num_winsRandom = 0

for itt in range(200):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = eval_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])

    if time_step.rewards[0] == 1:
        num_winsDQN += 1
    if time_step.rewards[1] == 1:
        num_winsRandom += 1

print("wins DQN: ", num_winsDQN)
print("wins Random: ", num_winsRandom)



agent_output = dqn_agent.step(time_step)
print(agent_output.action)
print(agent_output.probs)
print("amount of probs", len(agent_output.probs))
print("max prob ", max(agent_output.probs))
print("index of max prob", np.argmax(agent_output.probs))
time_step = env.step([agent_output.action])
time_step = env.step([agent_output.action + 1])
agent_output = dqn_agent.step(time_step)

print(agent_output.action)
print(agent_output.probs)
print("amount of probs", len(agent_output.probs))
print("max prob ", max(agent_output.probs))
print("index of max prob", np.argmax(agent_output.probs))







state = game.new_initial_state()
print(game.get_parameters())
print(state.get_game().get_parameters()["num_cols"])
print(state.get_game().get_parameters()["num_rows"])



print(game)
state = game.new_initial_state()
print(state)
print(game.num_players())
print(game.max_utility())
print(game.min_utility())
print(game.num_distinct_actions())
print(state.current_player())
print(state.is_terminal())
print(state.returns())
print(state.legal_actions())

state.apply_action(0)
print(state)
print(game.reward_model)
print(state.ob)
"""