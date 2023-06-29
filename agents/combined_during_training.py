import os
import tempfile

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np
import base64
import io

def load_weights_from_file(from_file_path):
    with open(from_file_path, "rb") as file:
        deep_q_network = base64.b64encode(file.read()).decode("utf-8")
    return deep_q_network


def save_weights_to_file(deep_q_network, to_file_path):
    with open(to_file_path, "w") as file:
        file.write(deep_q_network)


def encode_weights(from_file_path, to_file_path):
    encoded_weights = load_weights_from_file(from_file_path)
    save_weights_to_file(encoded_weights, to_file_path)



def create_model(configuration):
    input_layer = Input(shape=(configuration['rows'], configuration['columns'], 1))

    assert configuration['inarow'] % 2 == 0, "inarow must be even, for square the convolution layer to work"
    half_inarow = int(configuration['inarow'] / 2)
    conv_square_a = Conv2D(filters=32, kernel_size=(half_inarow + 1, half_inarow + 1), padding='same',
                           activation='swish')(input_layer)
    conv_square_b = Conv2D(filters=64, kernel_size=(half_inarow + 1, half_inarow + 1), padding='same',
                           activation='swish')(conv_square_a)
    conv_square_c = Conv2D(filters=128, kernel_size=(configuration['rows'], configuration['columns']), activation='swish')(conv_square_b)
    conv_square_d = Flatten()(conv_square_c)

    merged = Dense(128, activation='swish')(conv_square_d)

    value_layer = Dense(configuration['columns'], activation='linear')(merged)
    advantage_layer = Dense(configuration['columns'], activation='linear')(merged)

    advantage_delta = Lambda(lambda x: x - K.mean(x, axis=1, keepdims=True))(advantage_layer)
    output_layer = Lambda(lambda x: x[0] + x[1])([value_layer, advantage_delta])

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.build((None, configuration['rows'], configuration['columns'], 1))
    return model


class DqnWrapper:
    dqn_network = None
    model_weights_io = None


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
def calc(context, matrix, mark, depth=1, max_depth=7):
    if depth > max_depth:
        return 0, None

    matrix_hash = encode_matrix(matrix)

    if context.used[matrix_hash]:
        return context.res[matrix_hash]
    context.used[matrix_hash] = True

    is_all_losing = True
    no_valid_move = True
    non_losing_moves = []

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

        opponent_res, opponent_action = calc(context, next_matrix, 3 - mark, depth + 1, max_depth=max_depth)
        # If the opponent loses in all cases, then we'll take that move.
        if opponent_res == -1:
            context.res[matrix_hash] = (1, i)
            return context.res[matrix_hash]

        # If we found a neutral move for the opponent, then we'll take it as a neutral move for us
        if opponent_res == 0:
            is_all_losing = False
            non_losing_moves.append(i)

    if no_valid_move:
        context.res[matrix_hash] = (0, None)
    elif is_all_losing:
        context.res[matrix_hash] = (-1, None)
    else:
        context.res[matrix_hash] = (0, non_losing_moves)

    return context.res[matrix_hash]


# This agent looks 6 steps forward and chooses the step that leads to the winning, or NOT losing position
def brute_force_agent(observation, configuration, max_depth=7):
    context = Context(
        n=configuration.rows,
        m=configuration.columns,
        inrow=configuration.inarow,
        used=defaultdict(lambda: False),
        res=defaultdict(lambda: (0, 0)))

    board = observation['board']
    matrix = to_matrix(context, board)
    mark = observation.mark

    res, action = calc(context, matrix, mark, max_depth=max_depth)
    return res, action

def calculate_max_depth(observation, configuration):
    # Calculate number of empty cells in each column and sort in descending order
    board = observation['board']
    cnt = [0 for _ in range(configuration['columns'])]
    for j in range(configuration['columns']):
        for i in range(configuration['rows']):
            if board[i * configuration['columns'] + j] == 0:
                cnt[j] += 1
    cnt.sort(reverse=True)
    while len(cnt) > 0 and cnt[-1] == 0:
        # Remove from the end all columns that are full
        cnt.pop()

    depth = 0
    cnt_moves = 1
    while len(cnt) > 0 and cnt_moves < int(1 * 10 ** 5):
        depth += 1
        cnt_moves *= len(cnt)
        cnt[0] -= 1
        cnt.sort(reverse=True)
        while len(cnt) > 0 and cnt[-1] == 0:
            cnt.pop()

    return max(6, depth)

def combined_during_training_agent(observation, configuration):
    # Load the deep Q-network
    if DqnWrapper.dqn_network is None:
        DqnWrapper.dqn_network = create_model(configuration)
        DqnWrapper.model_weights_io = io.BytesIO(base64.b64decode(model_weights))

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(DqnWrapper.model_weights_io.read())
            tmp.flush()
        DqnWrapper.dqn_network.load_weights(tmp.name)
        os.unlink(tmp.name)

    # Get the action
    action = DqnWrapper.dqn_network.predict(
        np.array(observation['board'])
        .reshape((1, configuration['rows'], configuration['columns'], 1)))[0]

    max_depth = calculate_max_depth(observation, configuration)
    print("Calculated Max Depth: {}".format(max_depth))
    res_, action_ = brute_force_agent(observation, configuration, max_depth)

    if res_ == 1:
        return action_

    if res_ == 0:
        for i in range(configuration['columns']):
            if i not in action_:
                action[i] = -1000000

    for i in range(configuration['columns']):
        if observation['board'][i] != 0:
            action[i] = -1000000

    return int(np.argmax(action))