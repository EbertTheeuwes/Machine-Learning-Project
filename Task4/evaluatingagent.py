import pyspiel
import numpy as np
import random
import matplotlib.pyplot as plt
from open_spiel.python.egt import dynamics
from open_spiel.python import rl_environment
import dotsandboxes_agent
import time

game_string = "dots_and_boxes(num_rows=7,num_cols=7)"

print("Creating game: {}".format(game_string))
game = pyspiel.load_game(game_string) 
env = rl_environment.Environment(game)
num_players = env.num_players

num_actions = env.action_spec()["num_actions"]

start_load=time.perf_counter()
test_agent= dotsandboxes_agent.get_agent_for_tournament(0)
end_load = time.perf_counter()
elapsed_loadtime = end_load - start_load
print(f"Elapsed time: {elapsed_loadtime:.4f} seconds")  

eval_episodes=10
score=np.empty(eval_episodes)

start_time = time.perf_counter()
for cur_eval in range(eval_episodes):
    state = game.new_initial_state()
    test_agent.restart_at(state)
    while not state.is_terminal():
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        if current_player==1:
            rand_idx = random.randint(0, len(legal_actions) - 1)
            action = legal_actions[rand_idx]
        else:
            action=test_agent.step(state.clone())
        test_agent.inform_action(state,current_player,action)
        print(action)
        state.apply_action(action)
        
    returns = state.returns()
    if returns[0]>0:
       score[cur_eval]=1
    else:
       score[cur_eval]=0
    print("------------")
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")  

print(np.sum(score)/eval_episodes)