#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""
# simple MCTS agent


import sys
import math
import argparse
import logging
import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms import mcts



logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')

#MCTS parameters
uct_c=math.sqrt(2)
max_simulations=30
solve=True
num_playouts=3


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


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
        self.rng = np.random.RandomState()
        self.evaluator = mcts.RandomRolloutEvaluator(num_playouts, self.rng)
        self.initbot=False
        self.mcts_agent=""
        

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True
        self.mcts_agent.inform_action(state, player_id, action)

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        if self.initbot==False:
            self.mcts_agent= mcts.MCTSBot(state.get_game(),uct_c,max_simulations, self.evaluator,solve,random_state=self.rng,verbose=False)
            self.initbot=True
    
        action = self.mcts_agent.step(state)
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=5,num_cols=5)")
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

