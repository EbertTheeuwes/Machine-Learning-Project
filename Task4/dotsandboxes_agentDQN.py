#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import random
import numpy as np
import pyspiel
import torch
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.pytorch import dqn
from open_spiel.python import rl_environment



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


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        # BELANGRIJK: gaan er van uit dat telkens een nieuwe game gestart wordt, niewe agent gemaakt
        # wordt en dus deze init terug gecalled wordt, anders nog op bepaalde manier voor zorgen dat
        # rl_environment gereset wordt.
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.env = rl_environment.Environment("dots_and_boxes(num_rows=7,num_cols=7)")
        # self.num_players = self.env.num_players
        self.num_rows = 7
        self.num_columns = 7
        self.num_actions = self.env.action_spec()["num_actions"]
        self.info_state_size = self.env.observation_spec()["info_state"][0]
        self.dqn_agent = dqn.DQN(player_id=player_id, num_actions=self.num_actions, state_representation_size=self.info_state_size)
        model = torch.load("/Users/maxsebrechts/Desktop/Master/MLproj/Machine-Learning-Project/Task4/q_network_7x7.pt")
        model.eval()
        self.dqn_agent._q_network = model

        self.cur_time_step = self.env.reset()
        self.ILLEGAL_ACTION_VALUE = -np.inf


    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        num_cols = state.get_game().get_parameters()["num_cols"]
        num_rows = state.get_game().get_parameters()["num_rows"]
        trans_action = self.action_transform(action, num_rows, num_cols)
        #print("ACTION WHERE ERROR OCCURS 5x5", action)
        #print("ACTION WHERE ERROR OCCURS 7x7", trans_action)
        self.cur_time_step = self.env.step([trans_action])
        #print("state 7x7")
        #print(self.env.get_state)



    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        agent_output = self.dqn_agent.step(self.cur_time_step, is_evaluation=True)
        info_state = torch.Tensor(self.cur_time_step.observations["info_state"])
        q_vals = self.dqn_agent._q_network(info_state).detach()[0]
        #print(probs)

        indices_valid_actions = []
        first_vertical_action_orig = self.num_columns * (self.num_rows + 1)
        num_rows = state.get_game().get_parameters()["num_rows"]
        num_cols = state.get_game().get_parameters()["num_cols"]
        for i in range(num_rows+1):
            for j in range(num_cols):
                indices_valid_actions += [j + i*self.num_columns]

        for i in range(num_rows):
            for j in range(num_cols + 1):
                indices_valid_actions += [first_vertical_action_orig + j + i * (self.num_columns + 1)]

        legal_actions = self.env.get_state.legal_actions()
        #print("legal_actions : ", legal_actions)
        #print("indices_valid_actions ", indices_valid_actions)
        valid_probs = []
        for i in indices_valid_actions:
            if i in legal_actions:
                valid_probs += [q_vals[i]]
            else:
                valid_probs += [self.ILLEGAL_ACTION_VALUE]



        # as indices are added in proper order to indices_valid_actions, if probs are added in this order
        # to valid_probs, the order of the probs should correspond to how they would be ordered for
        # the corresponding smaller game. The index of a prob should therefore correspond to its action
        #print(valid_probs)
        action = np.argmax(valid_probs)
        #print("ACTION WHERE ERROR OCCURS, before transform", action)
        #print("ACTION WHERE ERROR OCCURS, in 7x7", self.action_transform(action, num_rows, num_cols))
        self.cur_time_step = self.env.step([self.action_transform(action, num_rows, num_cols)])
        #print("state 7x7")
        #print(self.env.get_state)
        return action



    def action_transform(self, action, num_rows, num_cols):
        """

        :param num_cols: number of columns in grid of game
        :param num_rows: number of rows in grid of game
        :param action: action in grid of game
        :return: corresponding action in 7x7 board
        """
        first_horizontal_action = (num_cols * (num_rows + 1))
        first_horizontal_action_orig = self.num_columns * (self.num_rows + 1)
        if action < first_horizontal_action:
            column = action % num_cols
            row = action // num_cols
            trans_action = row * self.num_columns + column
        else:
            action2 = action - first_horizontal_action
            column = action2 % (num_cols + 1)
            row = action2 // (num_cols + 1)
            first_horizontal_action_orig = self.num_columns * (self.num_rows + 1)
            trans_action = first_horizontal_action_orig + row * (self.num_columns + 1) + column

        return trans_action



def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dotsandboxes(num_rows=5,num_cols=5)")
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
