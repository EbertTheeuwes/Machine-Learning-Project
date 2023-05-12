# initialization
import pyspiel
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
# from mpltern.datasets import get_triangular_grid

# Define games
biased_rock_paper_scissors = pyspiel.create_matrix_game("brps", "biased_rock_paper_scissors", ["R", "P", "S"],
                                                        ["R", "P", "S"],
                                                        [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],
                                                        [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]])
dispersion = pyspiel.create_matrix_game("d", "dispersion", ["A", "B"], ["A", "B"], [[-1, 1], [1, -1]],
                                        [[-1, 1], [1, -1]])
battle_of_the_sexes = pyspiel.create_matrix_game("bots", "battle_of_the_sexes", ["O", "M"], ["O", "M"],
                                                 [[3, 0], [0, 2]], [[2, 0], [0, 3]])
prisoner_dilemma = pyspiel.create_matrix_game("pd", "prisoner_dilemma", ["C", "D"], ["C", "D"], [[-1, -4], [0, -3]],
                                              [[-1, 0], [-4, -3]])


# Lenient Boltzmann
# Define Action class
class Actions:
    def __init__(self, action, kappa, learningrate):
        self.action = action
        self.mean = 0
        self.N = 0
        self.kappa = kappa
        self.last_rewards = []
        self.learningrate = learningrate

    # Update the action-value estimate
    def update(self, x):
        self.last_rewards += [x]

        if len(self.last_rewards) == self.kappa:
            update_val = max(self.last_rewards)
            #print("updateval ", update_val)
            self.N += 1
            #print("n ", self.N)
            #self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * update_val
            self.mean = (1 - self.learningrate) * self.mean + self.learningrate * update_val
            #print("qval ", self.mean)
            self.last_rewards = []




# Perform boltzmann
def boltzmann(game, kappa, tau0, itt, learningrate):
    # Get actions
    state = game.new_initial_state()
    temp_actions_player_1 = state.legal_actions(0)
    temp_actions_player_2 = state.legal_actions(1)
    actions_player_1 = []
    actions_player_2 = []

    for action in temp_actions_player_1:
        actions_player_1 += [Actions(action, kappa, learningrate)]
    for action in temp_actions_player_2:
        actions_player_2 += [Actions(action, kappa, learningrate)]

    policies_player1 = np.empty(itt)
    policies_player2 = np.empty(itt)


    for i in range(itt):
        e_vals = []
        e_vals_sum = 0
        tau = tau0 - (tau0 - 0.01)/itt * i
        for action in actions_player_1:
            e_val = math.exp(action.mean/tau)
            e_vals += [e_val]
            e_vals_sum += e_val
            #print("eval ", e_val)

        prob_player1 = [e_val/e_vals_sum for e_val in e_vals]
        #print("probs ", prob_player1)

        e_vals = []
        e_vals_sum = 0
        for action in actions_player_2:
            e_val = math.exp(action.mean / tau)
            e_vals += [e_val]
            e_vals_sum += e_val

        prob_player2 = [e_val / e_vals_sum for e_val in e_vals]

        action1 = np.random.choice(actions_player_1, p=prob_player1)
        action2 = np.random.choice(actions_player_2, p=prob_player2)

        state = game.new_initial_state()
        state.apply_actions([action1.action, action2.action])
        results = state.returns()
        action1.update(results[0])
        action2.update(results[1])

        policies_player1[i] = prob_player1[0]
        policies_player2[i] = prob_player2[0]

    return [policies_player1, policies_player2]

"""
# battle of the sexes
game = battle_of_the_sexes
payoff_matrix = game_payoffs_array(game)
print("BATTLE OF THE SEXES")
print(payoff_matrix)
dyn = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)

# train learners
#c_1 = boltzmann(game, 5, 0.1, 100)
c_01 = boltzmann(game, 10, 0.1, 10000)
c_02 = boltzmann(game, 10, 0.2, 10000)
c_01L5 = boltzmann(game, 5, 0.1, 10000)



x_vals = np.linspace(0, 1, 20)
y_vals = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x_vals, y_vals)
# Computing dynamics/vector field
U = np.zeros_like(X)
V = np.zeros_like(Y)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        x = np.array([X[i,j],1-X[i,j],Y[i,j],1-Y[i,j]])
        x_dot = dyn(x)
        U[i,j] = x_dot[0]
        V[i,j] = x_dot[2]

# Plot the vector field
fig, ax = plt.subplots()
ax.quiver(X,Y,U,V)
#ax.plot(c_1[0],c_1[1],'b',label="eps=0.1")
# ax.plot(c_1[0][0],c_1[1][0],'bo',markersize=20)
# ax.plot(c_1[0][len(c_1[0])-1],c_1[1][len(c_1[1])-1],'bx',markersize=12)
ax.plot(c_01[0],c_01[1],'r',label="tau=0.01 kappa=10")
ax.plot(c_02[0], c_02[1], 'y', label="tau=0.02 kappa=10")
ax.plot(c_01L5[0], c_01L5[1], 'b', label="tau=0.01 kappa=5")

print("Policy, probability of picking action 1, after training of first player for c_02: ", c_02[0][len(c_02[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_02: ", c_02[1][len(c_02[1]) - 1])
print("Policy, probability of picking action 1, after training of first player for c_01: ", c_01[0][len(c_01[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_01: ", c_01[1][len(c_01[1]) - 1])
print("Policy, probability of picking action 1, after training of first player for c_01L5: ", c_01L5[0][len(c_01L5[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_01L5: ", c_01L5[1][len(c_01L5[1]) - 1])
#print("full policy history player 1 : ", c_01[0])
#print("full policy history player 2 : ", c_01[1])
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.legend()
plt.show()


#dispersion
game = dispersion
payoff_matrix = game_payoffs_array(game)
print("DISPERSION GAME")
print(payoff_matrix)
dyn = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)

#train learners
c_01 = boltzmann(game, 10, 0.1, 100000)
c_02 = boltzmann(game, 10, 0.2, 100000) # leniency hier op 10 zorgt dat niet kunnen convergeren, te hoge leniency zorgt hier er voor dat
# agents gwn confused geraken, want bij deze game moeten ze uiteindelijk gwn 2 verschillende acties nemen, maakt niet uit wie welke neemt
# hoge leniency gecombineerd met hoge tau (dus meer exploratie) zorgt er voor dat uiteindelijk beide acties voor agents geupdated gaan worden
# in richting van 1, en dus niet geconvergeerd kan worden
c_01L5 = boltzmann(game, 5, 0.1, 100000)

x_vals = np.linspace(0, 1, 20)
y_vals = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x_vals, y_vals)
# Computing dynamics/vector field
U = np.zeros_like(X)
V = np.zeros_like(Y)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        x = np.array([X[i,j],1-X[i,j],Y[i,j],1-Y[i,j]])
        x_dot = dyn(x)
        U[i,j] = x_dot[0]
        V[i,j] = x_dot[2]

# Plot the vector field
fig, ax = plt.subplots()
ax.quiver(X,Y,U,V)
ax.plot(c_01[0], c_01[1], 'b', label="tau=0.1")
# ax.plot(c_1[0][0],c_1[1][0],'bo',markersize=20)
# ax.plot(c_1[0][len(c_1[0])-1],c_1[1][len(c_1[1])-1],'bx',markersize=12)
ax.plot(c_02[0],c_02[1],'r',label="tau=0.2")
ax.plot(c_01L5[0],c_01L5[1],'y',label="tau=0.1 kappa=5")

print("Policy, probability of picking action 1, after training of first player for c_02: ", c_02[0][len(c_02[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_02: ", c_02[1][len(c_02[1]) - 1])
print("Policy, probability of picking action 1, after training of first player for c_01: ", c_01[0][len(c_01[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_01: ", c_01[1][len(c_01[1]) - 1])
print("Policy, probability of picking action 1, after training of first player for c_01L5: ", c_01L5[0][len(c_01L5[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_01L5: ", c_01L5[1][len(c_01L5[1]) - 1])
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.legend()
plt.show()
"""

#prisoner dilemma
game = prisoner_dilemma
print("PRISONERS DILEMMA")
payoff_matrix = game_payoffs_array(game)
print(payoff_matrix)
dyn = dynamics.MultiPopulationDynamics(payoff_matrix, dynamics.replicator)

# train learners
#c_1 = boltzmann(game, 5, 0.1, 100)
c_01 = boltzmann(game, 1, 0.5, 100000, 0.001)
#c_02 = boltzmann(game, 1, 0.2, 100000, 0.001)
#c_01L5 = boltzmann(game, 5, 0.1, 100000, 0.001)

x_vals = np.linspace(0, 1, 20)
y_vals = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x_vals, y_vals)
# Computing dynamics/vector field
U = np.zeros_like(X)
V = np.zeros_like(Y)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        x = np.array([X[i,j],1-X[i,j],Y[i,j],1-Y[i,j]])
        x_dot = dyn(x)
        U[i,j] = x_dot[0]
        V[i,j] = x_dot[2]

# Plot the vector field
fig, ax = plt.subplots()
ax.quiver(X,Y,U,V)
#ax.plot(c_1[0],c_1[1],'b',label="eps=0.1")
# ax.plot(c_1[0][0],c_1[1][0],'bo',markersize=20)
# ax.plot(c_1[0][len(c_1[0])-1],c_1[1][len(c_1[1])-1],'bx',markersize=12)
ax.plot(c_01[0],c_01[1],'b',label="tau=0.01 kappa=10")
#ax.plot(c_02[0], c_02[1], 'r', label="tau=0.02 kappa=10")
#ax.plot(c_01L5[0], c_01L5[1], 'y', label="tau=0.01 kappa=5")

#print("Policy, probability of picking action 1, after training of first player for c_02: ", c_02[0][len(c_02[0]) - 1])
#print("Policy, probability of picking action 1, after training of second player for c_02: ", c_02[1][len(c_02[1]) - 1])
print("Policy, probability of picking action 1, after training of first player for c_01: ", c_01[0][len(c_01[0]) - 1])
print("Policy, probability of picking action 1, after training of second player for c_01: ", c_01[1][len(c_01[1]) - 1])
#print("Policy, probability of picking action 1, after training of first player for c_01L5: ", c_01L5[0][len(c_01L5[0]) - 1])
#print("Policy, probability of picking action 1, after training of second player for c_01L5: ", c_01L5[1][len(c_01L5[1]) - 1])
#print("full policy history player 1 : ", c_01[0])
#print("full policy history player 2 : ", c_01[1])
ax.set_xlabel("Player 1")
ax.set_ylabel("Player 2")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.legend()
plt.show()
