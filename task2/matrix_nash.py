# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Find Nash equilibria for constant- or general-sum 2-player games.

Non-matrix games are handled by computing the normal (bimatrix) form.

The algorithms used are:
* direct computation of pure equilibria.
* linear programming to find equilibria for constant-sum games.
* iterated dominance to reduce the action space.
* reverse search vertex enumeration (if using lrsnash) to find all general-sum
  equilibria.
* support enumeration (if using nashpy) to find all general-sum equilibria.
* Lemke-Howson enumeration (if using nashpy) to find one general-sum
  equilibrium.

The general-sum mixed-equilibrium algorithms are likely to work well for tens of
actions, but less likely to scale beyond that.


Example usage:
```
matrix_nash_example --game kuhn_poker
```
"""


import itertools

from absl import app
from absl import flags
import nashpy
import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import matrix_nash
from open_spiel.python.egt import utils
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "first_sealed_auction(max_value=6)",
                    "Game (short name plus optional parameters).")
flags.DEFINE_float("tol", 1e-7, "Tolerance for determining dominance.")
flags.DEFINE_enum(
    "mode", "all", ["all", "pure", "one"], "Whether to find all extreme "
    "equilibria, all pure equilibria, or just one equilibrium.")
flags.DEFINE_enum(
    "solver", "nashpy", ["nashpy", "lrsnash", "linear"],
    "Solver to use for finding mixed equilibria. (lrsnash needs to"
    " be installed separately to work.)")
flags.DEFINE_string("lrsnash_path", None,
                    "Full path to lrsnash solver (searches PATH by default).")
flags.DEFINE_integer(
    "lrsnash_max_denom", 1000, "Maximum denominator to use "
    "when converting payoffs to rationals for lrsnash solver.")


def main(_):
  # Define games
  biased_rock_paper_scissors = pyspiel.create_matrix_game("brps", "biased_rock_paper_scissors", ["R", "P", "S"], ["R", "P", "S"], [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]], [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]])
  dispersion = pyspiel.create_matrix_game("d", "dispersion", ["A", "B"], ["A", "B"], [[-1, 1], [1, -1]], [[-1, 1], [1, -1]])
  battle_of_the_sexes = pyspiel.create_matrix_game("bots", "battle_of_the_sexes", ["O", "M"], ["O", "M"], [[3, 0], [0, 2]], [[2, 0], [0, 3]])
  prisoner_dilemma = pyspiel.create_matrix_game("pd", "prisoner_dilemma", ["C", "D"], ["C", "D"], [[-1, -4], [0, -3]], [[-1, 0], [-4, -3]])

# Choose game
  game = biased_rock_paper_scissors  
  #game = pyspiel.load_game(FLAGS.game)
  print("loaded game")

  # convert game to matrix form if it isn't already a matrix game
  if not isinstance(game, pyspiel.MatrixGame):
    game = pyspiel.extensive_to_matrix_game(game)
    num_rows, num_cols = game.num_rows(), game.num_cols()
    print("converted to matrix form with shape (%d, %d)" % (num_rows, num_cols))

  # use iterated dominance to reduce the space unless the solver is LP (fast)
  if FLAGS.solver != "linear":
    if FLAGS.mode == "all":
      game, _ = lp_solver.iterated_dominance(
          game, tol=FLAGS.tol, mode=lp_solver.DOMINANCE_STRICT)
      num_rows, num_cols = game.num_rows(), game.num_cols()
      print("discarded strictly dominated actions yielding shape (%d, %d)" %
            (num_rows, num_cols))
    if FLAGS.mode == "one":
      game, _ = lp_solver.iterated_dominance(
          game, tol=FLAGS.tol, mode=lp_solver.DOMINANCE_VERY_WEAK)
      num_rows, num_cols = game.num_rows(), game.num_cols()
      print("discarded very weakly dominated actions yielding shape (%d, %d)" %
            (num_rows, num_cols))

  # game is now finalized
  num_rows, num_cols = game.num_rows(), game.num_cols()
  row_actions = [game.row_action_name(row) for row in range(num_rows)]
  col_actions = [game.col_action_name(col) for col in range(num_cols)]
  row_payoffs, col_payoffs = utils.game_payoffs_array(game)
  pure_nash = list(
      zip(*((row_payoffs >= row_payoffs.max(0, keepdims=True) - FLAGS.tol)
            & (col_payoffs >= col_payoffs.max(1, keepdims=True) - FLAGS.tol)
           ).nonzero()))
  if pure_nash:
    print("found %d pure equilibria" % len(pure_nash))
  if FLAGS.mode == "pure":
    if not pure_nash:
      print("found no pure equilibria")
      return
    print("pure equilibria:")
    for row, col in pure_nash:
      print("payoffs %f, %f:" % (row_payoffs[row, col], col_payoffs[row, col]))
      print("row action:")
      print(row_actions[row])
      print("col action:")
      print(col_actions[col])
      print("")
    return
  if FLAGS.mode == "one" and pure_nash:
    print("pure equilibrium:")
    row, col = pure_nash[0]
    print("payoffs %f, %f:" % (row_payoffs[row, col], col_payoffs[row, col]))
    print("row action:")
    print(row_actions[row])
    print("col action:")
    print(col_actions[col])
    print("")
    return
  for row, action in enumerate(row_actions):
    print("row action %s:" % row)
    print(action)
  print("--")
  for col, action in enumerate(col_actions):
    print("col action %s:" % col)
    print(action)
  print("--")
  if num_rows == 1 or num_cols == 1:
    equilibria = itertools.product(np.eye(num_rows), np.eye(num_cols))
  elif FLAGS.solver == "linear":
    if FLAGS.mode != "one" or (row_payoffs + col_payoffs).max() > (
        row_payoffs + col_payoffs).min() + FLAGS.tol:
      raise ValueError("can't use linear solver for non-constant-sum game or "
                       "for finding all optima!")
    print("using linear solver")

    def gen():
      p0_sol, p1_sol, _, _ = lp_solver.solve_zero_sum_matrix_game(
          pyspiel.create_matrix_game(row_payoffs - col_payoffs,
                                     col_payoffs - row_payoffs))
      yield (np.squeeze(p0_sol, 1), np.squeeze(p1_sol, 1))

    equilibria = gen()
  elif FLAGS.solver == "lrsnash":
    print("using lrsnash solver")
    equilibria = matrix_nash.lrs_solve(row_payoffs, col_payoffs,
                                       FLAGS.lrsnash_max_denom,
                                       FLAGS.lrsnash_path)
  elif FLAGS.solver == "nashpy":
    if FLAGS.mode == "all":
      print("using nashpy vertex enumeration")
      equilibria = nashpy.Game(row_payoffs, col_payoffs).vertex_enumeration()
    else:
      print("using nashpy Lemke-Howson solver")
      equilibria = matrix_nash.lemke_howson_solve(row_payoffs, col_payoffs)
  print("equilibria:" if FLAGS.mode == "all" else "an equilibrium:")
  equilibria = iter(equilibria)
  # check that there's at least one equilibrium
  try:
    equilibria = itertools.chain([next(equilibria)], equilibria)
  except StopIteration:
    print("not found!")
  for row_mixture, col_mixture in equilibria:
    print("payoffs %f, %f for %s, %s" %
          (row_mixture.dot(row_payoffs.dot(col_mixture)),
           row_mixture.dot(
               col_payoffs.dot(col_mixture)), row_mixture, col_mixture))
    if FLAGS.mode == "one":
      return


if __name__ == "__main__":
  app.run(main)
