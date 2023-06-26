from collections import defaultdict
import copy


class Context:
    n = None
    m = None
    inrow = None
    used = None
    res = None

    def __init__(self, n, m, inrow, used, res):
        self.n = n
        self.m = m
        self.inrow = inrow
        self.used = used
        self.res = res


# Turns the list of cells into a matrix of a given size
def to_matrix(context, board):
    matrix = []
    for i in range(context.n):
        row = []
        for j in range(context.m):
            index = i * context.m + j
            row.append(board[index])
        matrix.append(row)
    return matrix


# Encodes the board into a unique number
def encode_matrix(matrix):
    matrix_hash = 0
    for row in matrix:
        for cell in row:
            matrix_hash = matrix_hash * 3 + cell
    return matrix_hash


def in_row_horizontal(context, matrix, x, y):
    for i in range(context.inrow):
        if y + i >= context.m or matrix[x][y] != matrix[x][y + i]:
            return False
    return True


def in_row_vertical(context, matrix, x, y):
    for i in range(context.inrow):
        if x + i >= context.n or matrix[x][y] != matrix[x + i][y]:
            return False
    return True


def in_row_right_diagonal(context, matrix, x, y):
    for i in range(context.inrow):
        if x + i >= context.n or y + i >= context.m or matrix[x][y] != matrix[x + i][y + i]:
            return False
    return True


def in_row_left_diagonal(context, matrix, x, y):
    for i in range(context.inrow):
        if x + i >= context.n or y - i < 0 or matrix[x][y] != matrix[x + i][y - i]:
            return False
    return True


# Checks if the game is finished
def is_game_finished(context, matrix):
    for i in range(context.n):
        for j in range(context.m):
            if matrix[i][j] == 0:
                continue
            if in_row_horizontal(context, matrix, i, j):
                return True
            if in_row_vertical(context, matrix, i, j):
                return True
            if in_row_right_diagonal(context, matrix, i, j):
                return True
            if in_row_left_diagonal(context, matrix, i, j):
                return True
    return False


# Generates a new matrix where the current move is applied
def get_next_matrix(context, matrix, col_index, mark):
    row_index = 0
    while row_index + 1 < context.n and matrix[row_index + 1][col_index] == 0:
        row_index += 1
    result = copy.deepcopy(matrix)
    result[row_index][col_index] = mark
    return result


# Calculates the winning and losing moves for the current state
def calc(context, matrix, mark, depth=1):
    if depth > 6:
        return 0, 0

    matrix_hash = encode_matrix(matrix)

    if context.used[matrix_hash]:
        return context.res[matrix_hash]
    context.used[matrix_hash] = True

    is_all_losing = True
    no_valid_move = True

    best_action = None
    for i in range(context.m):
        if matrix[0][i] != 0:
            continue
        else:
            no_valid_move = False
            if best_action is None:
                best_action = i

        next_matrix = get_next_matrix(context, matrix, i, mark)
        # If the game is finished, then we won
        if is_game_finished(context, next_matrix):
            context.res[matrix_hash] = (1, i)
            return context.res[matrix_hash]

        opponent_res, opponent_action = calc(context, next_matrix, 3 - mark, depth + 1)
        # If the opponent loses in all cases, then we'll take that move.
        if opponent_res == -1:
            context.res[matrix_hash] = (1, i)
            return context.res[matrix_hash]

        # If we found a neutral move for the opponent, then we'll take it as a neutral move for us
        if opponent_res == 0:
            is_all_losing = False
            best_action = i

    if no_valid_move:
        context.res[matrix_hash] = (0, -1)
    elif is_all_losing:
        context.res[matrix_hash] = (-1, best_action)
    else:
        context.res[matrix_hash] = (0, best_action)

    return context.res[matrix_hash]


# This agent looks 6 steps forward and chooses the step that leads to the winning, or NOT losing position
def brute_force_agent(observation, configuration):
    context = Context(
        n=configuration.rows,
        m=configuration.columns,
        inrow=configuration.inarow,
        used=defaultdict(lambda: False),
        res=defaultdict(lambda: (0, 0)))

    board = observation.board
    matrix = to_matrix(context, board)
    mark = observation.mark

    res, action = calc(context, matrix, mark)
    return action
