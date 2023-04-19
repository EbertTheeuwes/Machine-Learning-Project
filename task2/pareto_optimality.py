import pyspiel
import numpy as np
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
payoff_matrix = game_payoffs_array(game)

# Find Pareto optimal outcomes
pareto_optimal_outcomes = []
for i in range(len(payoff_matrix)):
    for j in range(len(payoff_matrix[i])):
        is_pareto_optimal = True
        for k in range(len(payoff_matrix)):
            if k != i and payoff_matrix[k][j][0] >= payoff_matrix[i][j][0] and payoff_matrix[k][j][1] >= payoff_matrix[i][j][1]:
                is_pareto_optimal = False
                break
        if is_pareto_optimal:
            pareto_optimal_outcomes.append((i, j))

# Print the Pareto optimal outcomes
print("Pareto optimal outcomes:", pareto_optimal_outcomes)