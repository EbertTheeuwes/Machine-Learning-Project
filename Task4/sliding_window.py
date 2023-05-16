import pyspiel
import time
from absl import app
games_list = pyspiel.registered_names()
assert "dots_and_boxes" in games_list
game_string = "dots_and_boxes(num_rows=3,num_cols=3)"

#print("Creating game: {}".format(game_string))
#game = pyspiel.load_game(game_string)
#state = game.new_initial_state("100111101111100111101111100111101111100111101111")
#state = game.new_initial_state()

#string = state.dbn_string()
#print(state)
#print(string)



def sliding_window(state, window_rows=7, window_cols=7):
    num_cols = state.get_game().get_parameters()["num_cols"]
    num_rows = state.get_game().get_parameters()["num_rows"]
    string = state.dbn_string()
    result = []
    indices = []
    row_padding = window_cols*(window_rows - num_rows)
    col_padding = (window_cols + 1)*(window_rows - num_rows)
    padding = window_cols - num_cols
    if window_rows <= num_rows and window_cols <= num_cols:
        for j in range(num_rows - window_rows + 1):
            for k in range(num_cols - window_cols + 1):
                #print('ja')
                new_string = ''
                old_indices = []
                start_index_rows = j*num_cols + k
                start_index_cols = j*(num_cols + 1) + k
                for i in range(window_rows + 1):
                    new_string += string[start_index_rows + i*num_cols: start_index_rows + i*num_cols + window_cols]
                    old_indices += [item for item in range(start_index_rows + i*num_cols, start_index_rows + i*num_cols + window_cols)]
                start_cols = num_cols*(num_rows + 1)
                for i in range(window_rows):
                    new_string += string[start_index_cols + start_cols + i*(num_cols +1): start_index_cols + start_cols + i*(num_cols +1) + (window_cols + 1)]
                    old_indices += [item for item in range(start_index_cols + start_cols + i*(num_cols +1), start_index_cols + start_cols + i*(num_cols +1) + (window_cols + 1))]
                result += [new_string]
                indices += [old_indices]

    elif window_rows > num_rows and window_cols <= num_cols:
        for k in range(num_cols - window_cols + 1):
            #print('ja')
            new_string = ''
            old_indices = []
            start_index_rows = k
            start_index_cols = k
            for i in range(num_rows + 1):
                new_string += string[start_index_rows + i*num_cols: start_index_rows + i*num_cols + window_cols]
                old_indices += [item for item in range(start_index_rows + i*num_cols, start_index_rows + i*num_cols + window_cols)]
            new_string += row_padding*'0'
            old_indices += [None]*row_padding
            start_cols = num_cols*(num_rows + 1)
            for i in range(num_rows):
                new_string += string[start_index_cols + start_cols + i*(num_cols +1): start_index_cols + start_cols + i*(num_cols +1) + (window_cols + 1)]
                old_indices += [item for item in range(start_index_cols + start_cols + i*(num_cols +1), start_index_cols + start_cols + i*(num_cols +1) + (window_cols + 1))]
            new_string += col_padding*'0'
            old_indices += [None]*col_padding
            result += [new_string]
            indices += [old_indices]

    elif window_rows <= num_rows and window_cols > num_cols:
        for j in range(num_rows - window_rows + 1):
            #print('ja')
            new_string = ''
            old_indices = []
            start_index_rows = j*num_cols
            start_index_cols = j*(num_cols + 1)
            for i in range(window_rows + 1):
                new_string += string[start_index_rows + i*num_cols: start_index_rows + i*num_cols + num_cols]
                new_string += padding*'0'
                old_indices += [item for item in range(start_index_rows + i*num_cols, start_index_rows + i*num_cols + num_cols)]
                old_indices += [None]*padding
            start_cols = num_cols*(num_rows + 1)
            for i in range(window_rows):
                new_string += string[start_index_cols + start_cols + i*(num_cols +1): start_index_cols + start_cols + i*(num_cols +1) + (num_cols + 1)]
                new_string += padding*'0'
                old_indices += [item for item in range(start_index_cols + start_cols + i*(num_cols +1), start_index_cols + start_cols + i*(num_cols +1) + (num_cols + 1))]
                old_indices += [None]*padding
            result += [new_string]
            indices += [old_indices]

    else:
        new_string = ''
        old_indices = []
        for i in range(num_rows + 1):
            new_string += string[i*num_cols: i*num_cols + num_cols]
            new_string += padding*'0'
            old_indices += [item for item in range(i*num_cols, i*num_cols + num_cols)]
            old_indices += [None]*padding
        new_string += row_padding*'0'
        old_indices += [None]*row_padding
        start_cols = num_cols*(num_rows + 1)
        for i in range(num_rows):
            new_string += string[start_cols + i*(num_cols +1): start_cols + i*(num_cols +1) + (num_cols + 1)]
            new_string += padding*'0'
            old_indices += [item for item in range(start_cols + i*(num_cols +1), start_cols + i*(num_cols +1) + (num_cols + 1))]
            old_indices += [None]*padding
        new_string += col_padding*'0'
        old_indices += [None]*col_padding
        result += [new_string]
        indices += [old_indices]

    #print(result)
    #for i in range(len(result)):
    #    print(result[i])
    #    game = pyspiel.load_game("dots_and_boxes(num_rows={},num_cols={})".format(window_rows, window_cols))
    #    new_state = game.new_initial_state(result[i])
    #    print(new_state)
    #    print(indices[i])
    return (result, indices)




#print(sliding_window(state, 4, 4))