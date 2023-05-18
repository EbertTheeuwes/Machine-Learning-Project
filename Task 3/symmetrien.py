import pyspiel

# Universal
def vertical_sym(state, game):
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    string = state.dbn_string()
    half = (num_rows + 1)*num_cols
    new_string = ''
    for i in range(num_rows + 1):
        new_string += string[num_cols*i:num_cols*(i+1)][::-1]
    for i in range(num_rows):
        new_string += string[half + (num_cols+1)*i:half + (num_cols+1)*(i+1)][::-1]
    return str(new_string)

def horizontal_sym(state, game):
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    string = state.dbn_string()
    half = (num_rows + 1)*num_cols
    new_string = ''
    for i in range(num_rows + 1):
        new_string += string[num_cols*(num_rows-i):num_cols*(num_rows-i)+num_cols]
    for i in range(num_rows):
        new_string += string[half + (num_cols+1)*(num_rows-1-i):half + (num_cols+1)*(num_rows-1-i)+num_cols+1]
    return str(new_string)

def rotate180degrees(state, game):
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    string = state.dbn_string()
    half = (num_rows + 1)*num_cols
    new_string = string[:half][::-1]
    new_string += string[half:][::-1]
    return str(new_string)

# 2X2 grid
def sym2x2diagonal1(state):
    string = state.dbn_string()
    new_string = (string[11] + string[8] + string[10] + string[7] + string[9] + string[6] + string[5] + 
                  string[3] + string[1] + string[4] + string[2] + string[0])
    return str(new_string)

def sym2x2diagonal2(state):
    string = state.dbn_string()
    new_string = (string[6] + string[9] + string[7] + string[10] + string[8] + string[11] + string[0] + 
                  string[2] + string[4] + string[1] + string[3] + string[5])
    return str(new_string) 

def rotate90degrees2x2(state):
    string = state.dbn_string()
    new_string = (string[8] + string[11] + string[7] + string[10] + string[6] + string[9] + string[1] + 
                  string[3] + string[5] + string[0] + string[2] + string[4])
    return str(new_string)

def rotate270degrees2x2(state):
    game_string = "dots_and_boxes(num_rows=2,num_cols=2)"
    game = pyspiel.load_game(game_string)
    string = rotate180degrees(state, game)
    newstate = game.new_initial_state(string)
    return rotate90degrees2x2(newstate)

# 1X1 grid
def sym1x1diagonal1(state):
    string = state.dbn_string()
    new_string = (string[3] + string[2] + string[1] + string[0])
    return str(new_string)

def sym1x1diagonal2(state):
    string = state.dbn_string()
    new_string = (string[2] + string[3] + string[0] + string[1])
    return str(new_string)

def rotate90degrees1x1(state):
    string = state.dbn_string()
    new_string = (string[2] + string[3] + string[1] + string[0])
    return str(new_string)

def rotate270degrees1x1(state):
    game_string = "dots_and_boxes(num_rows=1,num_cols=1)"
    game = pyspiel.load_game(game_string)
    string = rotate180degrees(state, game)
    newstate = game.new_initial_state(string)
    return rotate90degrees1x1(newstate)