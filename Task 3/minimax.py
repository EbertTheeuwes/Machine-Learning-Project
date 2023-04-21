import pyspiel
import time
from absl import app

transposition_table = dict()

def vertical_sym(state, game):
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    string = state.dbn_string()
    half = len(string) // 2
    new_string = ''
    for i in range(num_rows + 1):
        new_string += string[num_cols*i:num_cols*(i+1)][::-1]
    for i in range(num_rows):
        new_string += string[half + (num_cols+1)*i:half + (num_cols+1)*(i+1)][::-1]
    #assert new_string == sym2x2vertical(state)
    return str(new_string)

def horizontal_sym(state, game):
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    string = state.dbn_string()
    half = len(string) // 2
    new_string = ''
    for i in range(num_rows + 1):
        new_string += string[num_cols*(num_rows-i):num_cols*(num_rows-i)+num_cols]
    for i in range(num_rows):
        new_string += string[half + (num_cols+1)*(num_rows-1-i):half + (num_cols+1)*(num_rows-1-i)+num_cols+1]
    #assert new_string == sym2x2horizontal(state)
    return str(new_string)

def sym2x2vertical(state):
    string = state.dbn_string()
    new_string = (string[1] + string[0] + string[3] + string[2] + string[5] + string[4] + string[8] + 
                  string[7] + string[6] + string[11] + string[10] + string[9])
    return str(new_string)

def sym2x2horizontal(state):
    string = state.dbn_string()
    new_string = (string[4] + string[5] + string[2] + string[3] + string[0] + string[1] + string[9] + 
                  string[10] + string[11] + string[6] + string[7] + string[8])
    return str(new_string)

def rotate90degrees2x2(state):
    string = state.dbn_string()
    new_string = (string[8] + string[11] + string[7] + string[10] + string[6] + string[9] + string[1] + 
                  string[3] + string[5] + string[0] + string[2] + string[4])
    return str(new_string)

def sym2x2diagonal(state):
    string = state.dbn_string()
    new_string = (string[5] + string[4] + string[3] + string[2] + string[1] + string[0] + string[11] + 
                  string[10] + string[9] + string[8] + string[7] + string[6])
    return str(new_string)

def get_owners(state):
    first_wins = 0
    second_wins = 0
    for s in str(state):
        if s == '1':
            first_wins += 1
        elif s == '2':
            second_wins += 1
    return (first_wins, second_wins)

def _minimax(state, maximizing_player_id, game):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """

    if state.is_terminal():
        return state.player_return(maximizing_player_id)
    
    # Kijk transposition table na
    if (str(state.dbn_string()), get_owners(state), str(state.current_player())) in transposition_table:
        #print("yes")
        return transposition_table[(str(state.dbn_string()), get_owners(state), str(state.current_player()))]
    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min
    values_children = [_minimax(state.child(action), maximizing_player_id, game) for action in state.legal_actions()]
    result = selection(values_children)
    owners = get_owners(state)
    transposition_table[(str(state.dbn_string()), owners, str(player))] = result
    transposition_table[(vertical_sym(state, game), owners, str(player))] = result
    transposition_table[(horizontal_sym(state, game), owners, str(player))] = result
    params = game.get_parameters()
    num_rows = params['num_rows']
    num_cols = params['num_cols']
    if num_rows == num_cols:
        transposition_table[(rotate90degrees2x2(state), owners, str(player))] = result
        transposition_table[(sym2x2diagonal(state), owners, str(player))] = result
    return result


def minimax_search(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    v = _minimax(
        state.clone(), maximizing_player_id, game)
    return v


def main(_):
    start = time.time()
    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = "dots_and_boxes(num_rows=2,num_cols=2)"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)

    value = minimax_search(game)

    if value == 0:
        print("It's a draw")
    else:
        winning_player = 1 if value == 1 else 2
        print(f"Player {winning_player} wins.")
    print(time.time() - start)


if __name__ == "__main__":
    app.run(main)