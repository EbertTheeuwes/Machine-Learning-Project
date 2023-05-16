#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import math
import sys
import argparse
import logging
import random
import numpy as np
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
import pyspiel
from open_spiel.python.algorithms import evaluate_bots, mcts
import time


uct_c=0.5
max_simulations=40
solve=False


logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player

class DQNEvaluator(mcts.Evaluator):
    def __init__(self, dqn_agent1):
        self.agent= dqn_agent1
        self.env=""
        self.init=False
        self.q_vals=""

    def evaluate(self, state):
        if self.init==False:
            self.env = rl_environment.Environment(state.get_game())
            self.env._state=state
            timestep =self.env.get_time_step()
            info_state = torch.Tensor(timestep.observations["info_state"])
            self.q_vals = self.agent._q_network(info_state).detach()[0]
            self.init=True
        self.env._state=state
        legal_actions=state.legal_actions()
        num_actions = [i for i in range(0, state.get_game().num_distinct_actions())]
        actions_done = [num for num in num_actions if num not in legal_actions]
        q_vals = self.q_vals.clone()
        for ac in actions_done:
            q_vals[ac]=0
        timestep =self.env.get_time_step()
        while not timestep.last():
            actiont=torch.argmax(q_vals)
            action = actiont.item()
            q_vals[action]=0
            timestep = self.env.step([action])
        reward= timestep.rewards
        return reward

   
    
    def prior(self, state):
        if self.init==False:
            self.env = rl_environment.Environment(state.get_game())
            self.env._state=state
            timestep =self.env.get_time_step()
            info_state = torch.Tensor(timestep.observations["info_state"])
            self.q_vals = self.agent._q_network(info_state).detach()[0]
            self.init=True
        self.env._state=state
        timestep =self.env.get_time_step()
        q_vals=self.q_vals.clone()
        legal_actions = state.legal_actions()
        for action in legal_actions:
            q_vals[action]=0
        best_action=torch.argmax(self.q_vals).item()
        return [(action, 1 if action == best_action else 0) for action in legal_actions]





class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        model_path = '/home/reinhout/documents/dots_and_boxes/savedqn/q_network.pt'
        self.my_player1 = dqn.DQN(player_id=0, num_actions=112, state_representation_size=576)
        mlp_model = torch.load(model_path, map_location=torch.device('cpu'))
        self.my_player1._q_network.load_state_dict(mlp_model.state_dict())
        self.dqn_evaluator = DQNEvaluator(self.my_player1)
        self.rng = np.random.RandomState()
        self.initbot=False
        self.mcts_agent=""

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.dqn_evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True
        
        
        
        

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.dqn_evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True
        self.mcts_agent.inform_action(state, player_id, action)

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        
    
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.dqn_evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True
        
        action = self.mcts_agent.step(state.clone())
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())

