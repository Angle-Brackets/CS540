import random
import numpy as np
import time
from copy import deepcopy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    #board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces) # Chooses player piece
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1] # Choose opponent piece
        self.depth_max = 2
        self.board = [[' ' for j in range(5)] for i in range(5)]

    def heuristic_game_value(self, state):
        # First we need to decide if we are in a terminal state
        score = self.game_value(state)

        if abs(score) == 1:
            return score

        # Now we can calculate the heuristic manually by finding the distance between the current player's color
        # and all the other pieces of the same color, and attempting to minimize it.
        currentLocation = list()
        oppLocation = list()
        distancesAI = list()
        distancesOpp = list()
        ai_score = 0
        opp_score = 0

        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == self.my_piece:
                    currentLocation.append((i, j))
                elif state[i][j] == self.opp:
                    oppLocation.append((i, j))

        for i in range(len(currentLocation)):
            for j in range(i, len(currentLocation)):
                x = abs(currentLocation[i][0] - currentLocation[j][0])
                y = abs(currentLocation[i][1] - currentLocation[j][1])
                distance = max(x, y)
                distancesAI.append(distance)

        for i in range(len(oppLocation)):
            for j in range(i, len(oppLocation)):
                x = abs(oppLocation[i][0] - oppLocation[j][0])
                y = abs(oppLocation[i][1] - oppLocation[j][1])
                distance = max(x, y)
                distancesOpp.append(distance)

        if len(distancesAI) > 0:
            ai_score = 1 - (sum(distancesAI)/len(distancesAI))/5
        if len(distancesOpp) > 0:
            opp_score = 1 - (sum(distancesOpp)/len(distancesOpp))/5

        if ai_score > opp_score:
            return ai_score
        elif opp_score > ai_score:
            return -opp_score
        else:
            return 0

    def succ(self, state, current_piece):
        new_states = []
        placed_pieces = dict() # Keyed by position, color.
        R = len(state) # Length of Row
        C = len(state[0]) # Length of Column

        piece_count = 0
        for i in range(R):
            for j in range(C):
                if state[i][j] in self.pieces:
                    piece_count += 1

        drop_phase = piece_count < 8

        if drop_phase:
            # Loop through the grid and determine if a spot is filled, if it isn't, add it as a new state.
            for i in range(R):
                for j in range(C):
                    if state[i][j] not in self.pieces:
                        state_copy = deepcopy(state)  # Deep copy of state
                        state_copy[i][j] = current_piece
                        new_states.append(state_copy)
        else:
            # If we are not in the drop phase, we need to first find the positions of the currently placed pieces.
            # Then we need to make sure we don't overwrite those and perform a swap.
            for i in range(R):
                for j in range(C):
                    if state[i][j] in self.pieces:
                        placed_pieces[(i, j)] = state[i][j]

            # Now we do the swap for the current player.
            for position, piece in placed_pieces.items():
                if piece == current_piece: # We only care about the current piece
                    # Now we generate all the possible states from swapping this piece w/ a blank adj area.
                    y, x = position
                    # Generate all valid adjacent positions
                    adj_coords = list(filter(lambda coord : 0 <= coord[0] < len(state) and 0 <= coord[1] < len(state[0]),
                                                [
                                                    (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                                                    (x - 1, y), (x + 1, y),
                                                    (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
                                                ]
                                             ))
                    for coord in adj_coords:
                        if state[coord[1]][coord[0]] == ' ':
                            state_copy = deepcopy(state)
                            state_copy[y][x], state_copy[coord[1]][coord[0]] = state_copy[coord[1]][coord[0]], state_copy[y][x]
                            new_states.append(state_copy)
        return new_states

    # Always starts as MAX as the AI is the "starting" player
    def max_value(self, state, depth):
        # The turn can be determined by the depth, by just doing depth % 2, if its 0 its AI, otherwise it's the opponent
        current_player = self.my_piece if depth % 2 == 0 else self.opp
        score = self.heuristic_game_value(state)
        if abs(score) == 1:
            return score
        elif depth == self.depth_max:
            return score
        elif current_player == self.my_piece:
            new_states = self.succ(state, self.my_piece)
            for s in new_states:
                score = max(score, self.max_value(s, depth + 1))
            return score
        else:
            new_states = self.succ(state, self.opp)
            for s in new_states:
                score = min(score, self.max_value(s, depth + 1))
            return score

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        count =  0
        for row in state:
            for piece in row:
                count = count + (piece in self.pieces)

        drop_phase = count < 8 # Sets the drop_phase.

        # MINIMAX ALGO
        start = time.time()
        opt = (-2, [])
        for successor in self.succ(state, self.my_piece):
            opt = max((self.max_value(successor, 0), successor), opt)

        all_positions = dict()
        move = []
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] in self.pieces:
                    all_positions[(i, j)] = state[i][j]

        # Now choose the optimal move...
        for i in range(len(opt[1])):
            for j in range(len(opt[1][0])):
                if (i, j) not in all_positions and opt[1][i][j] in self.pieces:
                    move.append((i, j))

        if not drop_phase:
            for pos in all_positions:
                if opt[1][pos[0]][pos[1]] == ' ':
                    move.append(pos)

        #print(f"Move took: {time.time() - start} s")
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # Generates all the diagonals possible.
        # <3 https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
        board = np.array(state)
        diags = [board[::-1, :].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
        diags.extend(board.diagonal(i) for i in range(board.shape[1] - 1, -board.shape[0], -1))
        diags = list(filter(lambda x: len(x) >= 4, [n.tolist() for n in diags]))

        # We check all the diagonals in one shot, for those that have arrays of length 5 we need to step through them
        # starting from index 0 to 0 + 3, then index 1 for 1 + 4.
        for diagonal in diags:
            if len(diagonal) > 4 and self.pieces[0] in diagonal or self.pieces[1] in diagonal:
                for i in range(0, len(diagonal) - 3):
                    # Check if everything in the list is the same.
                    board_state = set(diagonal[i:len(diagonal) + i - 1])
                    if len(board_state) == 1 and ' ' not in board_state:
                        return 1 if diagonal[0] == self.my_piece else -1
            elif self.pieces[0] in diagonal or self.pieces[1] in diagonal:
                # This is for diagonal arrays of len 4
                if len(set(diagonal)) == 1 and ' ' not in set(diagonal):
                    return 1 if diagonal[0] == self.my_piece else -1

        # Checking box wins via checking a 2x2 region whilst stepping through the board
        for i in range(len(board) - 1):
            for j in range(len(board[i]) - 1):
                # Generate box!
                box = [board[i][j], board[i + 1][j], board[i][j + 1], board[i + 1][j + 1]]
                if self.pieces[0] in box or self.pieces[1] in box:
                    if len(set(box)) == 1:
                        return 1 if box[0] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    wins = 0
    total_games = 100
    for i in range(total_games):
        try:
            ai = TeekoPlayer()
            piece_count = 0
            turn = 0

            # drop phase
            while piece_count < 8 and ai.game_value(ai.board) == 0:

                # get the player or AI's move
                if ai.my_piece == ai.pieces[turn]:
                    ai.print_board()
                    move = ai.make_move(ai.board)
                    ai.place_piece(move, ai.my_piece)
                    print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
                else:
                    move_made = False
                    ai.print_board()
                    print(ai.opp + "'s turn")

                    move = []
                    (row, col) = (random.randint(0, 4), random.randint(0, 4))
                    while not ai.board[row][col] == ' ':
                        (row, col) = (random.randint(0, 4), random.randint(0, 4))

                    # ensure the destination (row,col) tuple is at the beginning of the move list
                    move.insert(0, (row, col))
                    ai.opponent_move(move)
                    move_made = True

                    '''
                    while not move_made:
                        player_move = input("Move (e.g. B3): ")
                        while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                            player_move = input("Move (e.g. B3): ")
                        try:
                            ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                            move_made = True
                        except Exception as e:
                            print(e)
                    '''

                # update the game variables
                piece_count += 1
                turn += 1
                turn %= 2

            # move phase - can't have a winner until all 8 pieces are on the board
            while ai.game_value(ai.board) == 0:

                # get the player or AI's move
                if ai.my_piece == ai.pieces[turn]:
                    ai.print_board()
                    move = ai.make_move(ai.board)
                    ai.place_piece(move, ai.my_piece)
                    print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
                    print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
                else:
                    move_made = False
                    ai.print_board()
                    print(ai.opp + "'s turn")
                    while not move_made:
                        move = []

                        (row, col) = (random.randint(0, 4), random.randint(0, 4))
                        while not ai.board[row][col] == ai.opp:
                            (row, col) = (random.randint(0, 4), random.randint(0, 4))

                        succ = ai.succ(ai.board, ai.opp)
                        for i in range(len(succ)):
                            if succ[i][row][col] == ' ':
                                if row - 1 >= 0 and succ[i][row - 1][col] == ai.opp and ai.board[row - 1][col] == ' ':
                                    move.append((row - 1, col))
                                    break
                                elif row + 1 <= 4 and succ[i][row + 1][col] == ai.opp and ai.board[row + 1][col] == ' ':
                                    move.append((row + 1, col))
                                    break
                                elif col - 1 >= 0 and succ[i][row][col - 1] == ai.opp and ai.board[row][col - 1] == ' ':
                                    move.append((row, col - 1))
                                    break
                                elif col + 1 <= 4 and succ[i][row][col + 1] == ai.opp and ai.board[row][col + 1] == ' ':
                                    move.append((row, col + 1))
                                    break
                                elif row - 1 >= 0 and col - 1 >= 0 and succ[i][row - 1][col - 1] == ai.opp and \
                                        ai.board[row - 1][col - 1] == ' ':
                                    move.append((row - 1, col - 1))
                                    break
                                elif row + 1 <= 4 and col - 1 >= 0 and succ[i][row + 1][col - 1] == ai.opp and \
                                        ai.board[row + 1][col - 1] == ' ':
                                    move.append((row + 1, col - 1))
                                    break
                                elif row - 1 >= 0 and col + 1 <= 4 and succ[i][row - 1][col + 1] == ai.opp and \
                                        ai.board[row - 1][col + 1] == ' ':
                                    move.append((row - 1, col + 1))
                                    break
                                elif row + 1 <= 4 and col + 1 <= 4 and succ[i][row + 1][col + 1] == ai.opp and \
                                        ai.board[row + 1][col + 1] == ' ':
                                    move.append((row + 1, col + 1))
                                    break

                        move.append((row, col))
                        ai.opponent_move(move)
                        move_made = True
                        '''
                        move_from = input("Move from (e.g. B3): ")
                        while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                            move_from = input("Move from (e.g. B3): ")
                        move_to = input("Move to (e.g. B3): ")
                        while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                            move_to = input("Move to (e.g. B3): ")
        
                        try:
                            ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                              (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                            move_made = True
                        except Exception as e:
                            print(e)
                        '''

                # update the game variables
                turn += 1
                turn %= 2

            ai.print_board()
            if ai.game_value(ai.board) == 1:
                wins += 1
                print("AI wins! Game over.")
            else:
                print("You win! Game over.")
        except:
            pass
    print(f"Win percentage: {100*(wins/total_games)}%")

if __name__ == "__main__":
    main()
# def main():
#     print('Hello, this is Samaritan')
#     ai = TeekoPlayer()
#     piece_count = 0
#     turn = 0
#
#     # drop phase
#     while piece_count < 8 and ai.game_value(ai.board) == 0:
#
#         # get the player or AI's move
#         if ai.my_piece == ai.pieces[turn]:
#             ai.print_board()
#             move = ai.make_move(ai.board)
#             ai.place_piece(move, ai.my_piece)
#             print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
#         else:
#             move_made = False
#             ai.print_board()
#             print(ai.opp + "'s turn")
#             while not move_made:
#                 player_move = input("Move (e.g. B3): ")
#                 while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
#                     player_move = input("Move (e.g. B3): ")
#                 try:
#                     ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
#                     move_made = True
#                 except Exception as e:
#                     print(e)
#
#         # update the game variables
#         piece_count += 1
#         turn += 1
#         turn %= 2
#
#     # move phase - can't have a winner until all 8 pieces are on the board
#     while ai.game_value(ai.board) == 0:
#
#         # get the player or AI's move
#         if ai.my_piece == ai.pieces[turn]:
#             ai.print_board()
#             move = ai.make_move(ai.board)
#             ai.place_piece(move, ai.my_piece)
#             print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
#             print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
#         else:
#             move_made = False
#             ai.print_board()
#             print(ai.opp + "'s turn")
#             while not move_made:
#                 move_from = input("Move from (e.g. B3): ")
#                 while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
#                     move_from = input("Move from (e.g. B3): ")
#                 move_to = input("Move to (e.g. B3): ")
#                 while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
#                     move_to = input("Move to (e.g. B3): ")
#                 try:
#                     ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
#                                       (int(move_from[1]), ord(move_from[0]) - ord("A"))])
#                     move_made = True
#                 except Exception as e:
#                     print(e)
#
#         # update the game variables
#         turn += 1
#         turn %= 2
#
#     ai.print_board()
#     if ai.game_value(ai.board) == 1:
#         print("AI wins! Game over.")
#     else:
#         print("You win! Game over.")
#
#
# if __name__ == "__main__":
#     main()
