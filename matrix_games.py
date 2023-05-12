import sys
import pyspiel
import numpy as np
import random
import matplotlib.pyplot as plt
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner, boltzmann_tabular_qlearner
print("")

# Define games
biased_rock_paper_scissors = pyspiel.create_matrix_game("brps", "biased_rock_paper_scissors", ["R", "P", "S"], ["R", "P", "S"], [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]], [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]])
dispersion = pyspiel.create_matrix_game("d", "dispersion", ["A", "B"], ["A", "B"], [[-1, 1], [1, -1]], [[-1, 1], [1, -1]])
battle_of_the_sexes = pyspiel.create_matrix_game("bots", "battle_of_the_sexes", ["O", "M"], ["O", "M"], [[3, 0], [0, 2]], [[2, 0], [0, 3]])
prisoner_dilemma = pyspiel.create_matrix_game("pd", "prisoner_dilemma", ["C", "D"], ["C", "D"], [[-1, -4], [0, -3]], [[-1, 0], [-4, -3]])

# Choose game
game = battle_of_the_sexes
print(game)
state = game.new_initial_state()
print(state)  # action names also not provided; defaults used

# ε-greedy
# Define Action class
class Actions:
    def __init__(self, action):
        self.action = action
        self.mean = 0
        self.N = 0

    # Update the action-value estimate
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * x

# Get actions
temp_actions_player_1 = state.legal_actions(0)
temp_actions_player_2 = state.legal_actions(1)
actions_player_1 = []
actions_player_2 = []

for action in temp_actions_player_1:
    actions_player_1 += [Actions(action)]
for action in temp_actions_player_2:
    actions_player_2 += [Actions(action)]

# Perform ε-greedy
def epsilon(game, actions_player_1, actions_player_2, eps1, eps2, itt):
    data1 = np.empty(itt)
    data2 = np.empty(itt)
    start1 = True
    start2 = True
    for i in range(itt):
        number1 = np.random.random()
        number2 = np.random.random()
        if number1 < eps1 or start1:
            action1 = random.choice(actions_player_1)
        else:
            action1 = actions_player_1[np.argmax([act.mean for act in actions_player_1])]
        if number2 < eps2 or start2:
            action2 = random.choice(actions_player_2)
        else:
            action2 = actions_player_2[np.argmax([act.mean for act in actions_player_2])]
        start1 = False
        start2 = False

        state = game.new_initial_state()
        state.apply_actions([action1.action, action2.action])
        results = state.returns()
        action1.update(results[0])
        action2.update(results[1])

        # for the plot
        data1[i] = results[0]
        data2[i] = results[1]
    cumulative_average1 = np.cumsum(data1) / (np.arange(itt) + 1)
    cumulative_average2 = np.cumsum(data2) / (np.arange(itt) + 1)

    for action in actions_player_1:
        print(action.mean)
    for action in actions_player_2:
        print(action.mean)

    return [cumulative_average1, cumulative_average2]

# Perform reinforcement learning
if __name__ == '__main__':
    c_1 = epsilon(game, actions_player_1, actions_player_2, 0.1, 0.1, 100000)[0]
    c_05 = epsilon(game, actions_player_1, actions_player_2, 0.05, 0.05, 100000)[0]
    c_01 = epsilon(game, actions_player_1, actions_player_2, 0.01, 0.01, 100000)[0]


    # log scale plot
    plt.plot(c_1, label ='eps = 0.1')
    plt.plot(c_05, label ='eps = 0.05')
    plt.plot(c_01, label ='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.figure(figsize = (12, 8))
    plt.plot(c_1, label ='eps = 0.1')
    plt.plot(c_05, label ='eps = 0.05')
    plt.plot(c_01, label ='eps = 0.01')
    plt.legend()
    plt.show()
