"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def my_moves_score(game, player):
    return float(len(game.get_legal_moves(player)))


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Denoted 'force' as a metric not necessarily by the number of moves the
    # opponent has, but how close they are to the edge of the board. Knowing
    # the 'Knight'-style movement constraints the closer to the edge of the
    # board one is the fewer moves there will be not only on the current, but
    # future plays as well

    opponent_location = game.get_player_location(game.get_opponent(player))

    half_game_width = int(game.width / 2)
    half_game_height = int(game.height / 2)

    return float((half_game_width + half_game_height) - (
            abs(opponent_location[0] - half_game_width) + 
            abs(opponent_location[1] - half_game_height)))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        try:
            # Perform any required initializations, including selecting an
            # initial move from the game board (i.e., an opening book), or
            # returning immediately if there are no legal moves, all of which
            # should cost time as a human player would do this as well during
            # their alloted time

            # Some trickery to avoid an off-by-one error when accessing a
            # 0-index array
            middle_cell = (int(game.width / 2), int(game.height / 2))

            # Set the default best available move as random among the options
            # or (-1, -1) if nothing is available
            if legal_moves:
                best_move = random.choice(legal_moves)
            else:
                best_move = (-1, -1)

            # First check if we have the upper hand for an opening salvo ;)
            if game.move_count == 0:
                # The best first move is always the center location
                return middle_cell
            if game.move_count == 1:
                # The second move should get us closest into the center of the
                # board given the 'Knight'-style movement
                if game.move_is_legal(middle_cell):
                    return middle_cell
                else:
                    pass

            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.method == 'minimax':
                if self.iterative:
                    depth = 1
                    while True:  # we haven't timed out yet
                        score, best_move = self.minimax(game, depth)
                        depth += 1  # increase depth for each iteration
                else:
                    score, best_move = self.minimax(game, self.search_depth)
            elif self.method == 'alphabeta':
                if self.iterative:
                    depth = 1
                    while True:
                        score, best_move = self.alphabeta(game, depth)
                        depth += 1
                else:
                    score, best_move = self.alphabeta(game, self.search_depth)
            else:  # we've made it to an incorrectly spelled method type
                raise NameError()

        except Timeout:
            pass
            # Handle any actions required at timeout, if necessary
            #return best_move

        ## we dont want this # We should never get here, but if we do...
        # Return the best move from the last completed search iteration
        return best_move


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # leveraged help at: http://neverstopbuilding.com/minimax
        scores = []

        # determine which player we are
        if maximizing_player:
            whoami = game.active_player
        else:
            whoami = game.inactive_player

        # determine the moves of the other (active!) player
        legal_moves = game.get_legal_moves(game.active_player)

        # if no legal moves available, return the score immediately as someone
        # has thus won
        if not legal_moves:
            return self.score(game, whoami)

        # determine the depth from `depth` and compute out the tree
        for legal_move in legal_moves:
            forecasted_game = game.forecast_move(legal_move)
            if depth > 1: # if we need to recur
                scores.append([self.minimax(forecasted_game,
                                            depth - 1,
                                            not maximizing_player)[0],
                               legal_move])
            else: # we need to determine a score for the forecasted game board
                scores.append([self.score(forecasted_game, whoami), legal_move])

        # score the tree using minimax
        if maximizing_player:
            score = max(scores, key=lambda x: x[0])
        else: # minimizing player
            score = min(scores, key=lambda x: x[0])

        # return the score and the tuple(int,int) for position
        return score


    def alphabeta(self,
                  game,
                  depth,
                  alpha=float("-inf"),
                  beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # leveraged help at: http://neverstopbuilding.com/minimax
        scores = []

        # determine which player we are
        if maximizing_player:
            whoami = game.active_player
        else:
            whoami = game.inactive_player

        # determine the moves of the other (active!) player
        legal_moves = game.get_legal_moves(game.active_player)

        # if no legal moves available, return the score immediately as someone
        # has thus won
        if not legal_moves:
            return self.score(game, whoami)

        # determine the depth from `depth` and compute out the tree
        for legal_move in legal_moves:

            # if alpha ever greater than beta, prune and continue
            if alpha >= beta:
                continue

            forecasted_game = game.forecast_move(legal_move)

            if depth > 1:  # if we need to recur
                score, _ = self.alphabeta(forecasted_game,
                                          depth - 1,
                                          alpha=alpha,
                                          beta=beta,
                                          maximizing_player=not maximizing_player)
                # set alpha if the returned score is greater than the current
                # alpha value
                if maximizing_player and score > alpha:
                    alpha = score
                elif not maximizing_player and score < beta:  # likewise w/beta
                    beta = score

                scores.append([score, legal_move])
            else:  # we need to determine a score for the forecasted game board
                score = self.score(forecasted_game, whoami)

                # same as above here
                if maximizing_player and score > alpha:
                    alpha = score
                elif not maximizing_player and score < beta:
                    beta = score

                scores.append([score, legal_move])

        # score the tree using minimax
        if maximizing_player:
            score_and_move = max(scores, key=lambda x: x[0])
        else:  # minimizing player
            score_and_move = min(scores, key=lambda x: x[0])

        # return the score and the tuple(int,int) for position
        return score_and_move
