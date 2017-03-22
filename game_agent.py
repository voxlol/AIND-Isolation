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

def aggressive_heuristic(game, player):
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))

    return float(player_moves - opponent_moves**2)

def open_spaces_heuristic(game, player):
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))
    blank_spaces = len(game.get_blank_spaces())

    return float((player_moves - opponent_moves) / blank_spaces)

def square_player_heuristic(game, player):
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))

    return float(player_moves**2 - opponent_moves)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))
    
    blank_spaces = len(game.get_blank_spaces())
    total_possible_moves = game.width * game.height
    played_moves = total_possible_moves - blank_spaces

    return float(player_moves - 1.5*opponent_moves)

def game_completion_weight_heuristic(game, player):
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))
    
    blank_spaces = len(game.get_blank_spaces())
    total_possible_moves = game.width * game.height
    played_moves = total_possible_moves - blank_spaces

    percentage_game_completed = played_moves / total_possible_moves

    normalized_coef = percentage_game_completed * 2

    return float(player_moves - normalized_coef*opponent_moves) 

def moderately_aggressive_heuristic(game, player):
    opponent = game.get_opponent(player)
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(opponent))
    
    blank_spaces = len(game.get_blank_spaces())
    total_possible_moves = game.width * game.height
    played_moves = total_possible_moves - blank_spaces

    return float(player_moves - 1.5*opponent_moves)

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
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) == 0 or legal_moves == []:
            return (-1, -1)

        # Play center move if possible
        # center_move = (game.height // 2, game.width // 2)
        # if center_move in legal_moves:
            # return center_move

        # rotation hash
            # flip 90 degrees 4 times  
            # flip diagonally

        move = legal_moves[0]

        try:
            max_depth = game.width * game.height if self.iterative else self.search_depth
            for current_depth in range(1, max_depth + 1):
                if self.method == 'alphabeta':
                    points, move = self.alphabeta(game, current_depth)
                else:
                    points, move = self.minimax(game, current_depth)
        except Timeout:
            pass
            
        return move

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
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        current_score = self.score(game, self)
        moves = game.get_legal_moves(game.active_player)

        # terminate if game is over
        if game.is_winner(self) or game.is_loser(self) or depth == 0:
            return (current_score, game.get_player_location(self))
        # If we've reached a leaf node OR there are no legal moves, evaluate & return
        if len(moves) == 0:
            return current_score, (-1, -1)

        result = ()
        # Iterate through all the legal moves and calculate the backpropagated utility for each new game state
        for move in moves:
            new_state = game.forecast_move(move)    # create new game state by playing the legal move
            points, _ = self.minimax(new_state, depth-1, not maximizing_player) # recusively call minimax while decrementing the depth and fliping the maxmizing_player flag
            
            # Update the result tuple to maximize point and keep the move that maximizes the points
            if len(result) == 0:
                result = (points, move)
            elif maximizing_player and points > result[0]:
                result = (points, move)
            elif not maximizing_player and points < result[0]:
                result = (points, move)

        return result

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
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
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # If leaf node, return score and move
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # Get current legal moves for current player
        moves = game.get_legal_moves(game.active_player)

        if len(moves) == 0:
            return self.score(game, self), (-1, -1) 

        best_move = ()

        # Loop through all moves 
        for move in moves:
            # Exit this branch of search if we've already hit the alpha/beta bound
            if len(best_move) > 0:
                is_max_and_at_upper_bound = maximizing_player and best_move[0] >= beta
                is_min_and_at_lower_bound = not maximizing_player and best_move[0] <= alpha
                if is_max_and_at_upper_bound or is_min_and_at_lower_bound:
                    return best_move

            # Recursively search the next move
            new_game_state = game.forecast_move(move)
            move_points, _ = self.alphabeta(
                new_game_state,
                depth-1,
                alpha,
                beta,
                not maximizing_player)

            # Update running best_move
            if len(best_move) > 0:
                is_max_and_higher = maximizing_player and move_points > best_move[0]
                is_min_and_lower = not maximizing_player and move_points < best_move[0]
            if len(best_move) == 0 or is_max_and_higher or is_min_and_lower:
                best_move = (move_points, move)
          
            # Update alpha/beta
            if maximizing_player:
                alpha = best_move[0]
            else:
                beta = best_move[0]

        return best_move
