# Create a class called DQN, that has the following methods:
#   - __init__(env): initialize the DQN with the given environment.
#   - train(): train the DQN for a given number of episodes.
#   - create_model(): creates and returns the deep Q-network model.
#   - remember(state, action, reward, next_state, done): remember a state, action, reward, next state, and done.
#   - target(): calculate the target value for a given state
#   - act(state): return the best possible action for a given state.
from collections import deque
import numpy as np
import tensorflow as tf
from kaggle_environments import make
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Conv2D, Flatten, Concatenate
import tensorflow.keras.backend as K
import gc

from tensorflow.keras.optimizers.legacy import Adam
from scipy.special import softmax
from tabulate import tabulate

from agents.brute_force_agent import brute_force_agent

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
    if depth > 5:
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

        opponent_res, opponent_action = calc(context, next_matrix, 3 - mark, depth + 1)
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
def brute_force_agent(observation, configuration):
    context = Context(
        n=configuration.rows,
        m=configuration.columns,
        inrow=configuration.inarow,
        used=defaultdict(lambda: False),
        res=defaultdict(lambda: (0, 0)))

    board = observation['board']
    matrix = to_matrix(context, board)
    mark = observation.mark

    res, action = calc(context, matrix, mark)
    return res, action


class DQN:
    def __init__(self, env, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95, tau=0.001,
                 render_frequency=50, save_frequency=10, update_frequency=4, batch_size=64, learning_rate=0.001):
        self.env = env
        self.rows = self.env.configuration['rows']
        self.columns = self.env.configuration['columns']
        self.inarow = self.env.configuration['inarow']
        self.state_space = self.rows * self.columns
        self.action_space = self.columns

        self.memory = deque(maxlen=1000)
        self.positive_memory = deque(maxlen=1000)
        self.negative_memory = deque(maxlen=1000)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.learning_rate = learning_rate
        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.trainer = None

        self.render_frequency = render_frequency
        self.save_frequency = save_frequency
        self.update_frequency = update_frequency
        self.batch_size = batch_size

        self.total_steps = 0
        self.reward_per_episode = []

    def create_model(self):
        input_layer = Input(shape=(self.rows, self.columns, 1))

        assert self.inarow % 2 == 0, "inarow must be even, for square the convolution layer to work"
        half_inarow = int(self.inarow / 2)
        conv_square_a = Conv2D(filters=32, kernel_size=(half_inarow + 1, half_inarow + 1), padding='same', activation='swish')(input_layer)
        conv_square_b = Conv2D(filters=64, kernel_size=(half_inarow + 1, half_inarow + 1), padding='same', activation='swish')(conv_square_a)
        conv_square_c = Conv2D(filters=128, kernel_size=(self.rows, self.columns), activation='swish')(conv_square_b)
        conv_square_d = Flatten()(conv_square_c)

        merged = Dense(128, activation='swish')(conv_square_d)

        value_layer = Dense(self.columns, activation='linear')(merged)
        advantage_layer = Dense(self.columns, activation='linear')(merged)

        advantage_delta = Lambda(lambda x: x - K.mean(x, axis=1, keepdims=True))(advantage_layer)
        output_layer = Lambda(lambda x: x[0] + x[1])([value_layer, advantage_delta])

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        model.build((None, self.rows, self.columns, 1))
        return model

    def update_target_model(self):
        main_model_weights = self.main_model.get_weights()
        target_model_weights = self.target_model.get_weights()

        for i in range(len(main_model_weights)):
            target_model_weights[i] = main_model_weights[i] * self.tau + target_model_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_model_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([
            self.extract_state(state),
            action,
            reward,
            self.extract_state(next_state),
            done])

        if done:
            if reward > 0:
                # Copy all the memories to the positive memory
                self.positive_memory.extend(self.memory)
            else:
                self.negative_memory.extend(self.memory)
            self.memory.clear()

    def sample(self, batch_size):
        batch = []
        for i in range(batch_size):
            if i % 2 == 0:
                batch.append(self.positive_memory[np.random.randint(0, len(self.positive_memory))])
            else:
                batch.append(self.negative_memory[np.random.randint(0, len(self.negative_memory))])

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        for observation, action, reward, next_observation, done in batch:
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            dones.append(done)
        observations = np.array(observations).reshape((batch_size, self.rows, self.columns, 1))
        actions = np.array(actions).reshape((batch_size, 1))
        rewards = np.array(rewards).reshape((batch_size, 1))
        next_observations = np.array(next_observations).reshape((batch_size, self.rows, self.columns, 1))
        dones = np.array(dones).reshape((batch_size, 1))

        return observations, actions, rewards, next_observations, dones

    def target(self, state, model=None):
        if model is None:
            return self.main_model.predict(state, verbose=0)
        else:
            return model.predict(state, verbose=0)

    def visualize_thoughts(self, observation, configuration, action, reward):
        act_values = self.target(self.extract_state(observation))[0]
        probs = softmax(act_values)
        print("________________________")
        print("!!!!!!!!!!!!!!!!!!!!!!!")
        print("Board:")
        for row in np.reshape(self.extract_state(observation), (self.rows, self.columns)):
            [print("X" if x == 1 else "O" if x == -1 else "-", end=" ") for x in row]
            print()
        probs *= 100
        probs = np.round(probs)

        act_values = np.round(act_values, 2)
        table = np.vstack((act_values, probs))

        print(tabulate(table))

        print("Action: ", action)
        print("Reward: ", reward)

        print("!!!!!!!!!!!!!!!!!!!!!!!")
        print("________________________")

    def act(self, observation, configuration):
        res, action = brute_force_agent(observation, configuration)
        if res == 1:
            return action

        act_values = self.target(self.extract_state(observation))[0]

        if res == 0:
            for i in range(len(act_values)):
                if i not in action:
                    act_values[i] = -1000

        for i in range(len(act_values)):
            if observation['board'][i] != 0:
                act_values[i] = -1000

        probs = softmax(act_values)
        chosen_index = np.random.choice(np.arange(len(act_values)), p=probs)
        return int(chosen_index)

    def agent(self, observation, configuration):
        act_values = self.target(self.extract_state(observation))[0]
        for i, val in enumerate(act_values):
            if observation['board'][i] != 0:
                act_values[i] = -1000
        res, action = brute_force_agent(observation, configuration)
        if res == 1:
            return action
        if res == 0:
            for i in range(len(act_values)):
                if i not in action:
                    act_values[i] = -1000
        return int(np.argmax(act_values))

    def train(self, episodes):
        for episode in range(episodes):
            self.init_episode(episode)
            self.run_episode()

    def init_episode(self, episode):
        print("Initializing episode #{} ...".format(episode))
        if episode % self.render_frequency == 0:
            self.render(self.agent, "dqn.html")
            print("--------------------")
            print("--------------------")
            print("Episode finished with reward = {}".format(np.mean(np.array(self.reward_per_episode[-self.render_frequency:]))))
            print("--------------------")
            print("--------------------")
        if episode % self.save_frequency == 0:
            self.flush_memory(episode)
        opponent = ["negamax", "random"][np.random.randint(0, 2)]
        if np.random.rand() > 0.5:
            self.trainer = self.env.train([None, self.agent])
        else:
            self.trainer = self.env.train([self.agent, None])

    def run_episode(self):
        observation = self.trainer.reset()
        print("Maximum Q-Value from the beginning of the game: {}"
              .format(np.max(self.target(self.extract_state(observation), self.target_model)[0])))
        done = False
        while not done:
            action = self.act(observation, self.env.configuration)
            next_observation, reward, done, _ = self.trainer.step(action)

            if reward is None:
                reward = -1
            reward = max(reward * 8, 0)
            self.remember(observation, action, reward, next_observation, done)

            if done:
                self.reward_per_episode.append(reward)
                self.visualize_thoughts(observation, self.env.configuration, action, reward)

            self.total_steps += 1
            if self.total_steps % self.update_frequency == 0:
                self.update_main_model()
                self.update_target_model()

            observation = next_observation

    def update_main_model(self):
        if len(self.positive_memory) + len(self.negative_memory) < self.batch_size:
            return

        observations, actions, rewards, next_observations, dones = self.sample(self.batch_size)

        targets = rewards + self.gamma * np.max(self.target(next_observations, self.target_model), axis=1).reshape((self.batch_size, 1)) * (1.0 - dones)
        targets_full = self.target(observations, self.main_model)
        for i in range(self.batch_size):
            targets_full[i, actions[i]] = targets[i, 0]
        self.main_model.fit(observations, targets_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def flush_memory(self, episode):
        print("_________________________")
        print("Garbage collecting ... ")
        gc.collect()
        print("Savings model ... ")
        self.save("dqn-main.h5", "dqn-target.h5")
        print("Clearing memory ... ")
        tf.keras.backend.clear_session()
        print("Loading model ...")
        self.load("dqn-main.h5", "dqn-target.h5")
        print("Done!")
        print("_________________________")

    def save(self, main_model_file_path, target_model_file_path):
        self.main_model.save_weights(main_model_file_path)
        self.target_model.save_weights(target_model_file_path)

    def load(self, main_model_file_path, target_model_file_path):
        self.main_model.load_weights(main_model_file_path)
        self.target_model.load_weights(target_model_file_path)

    def render(self, agent, file_name="render.html"):
        print("Rendering ... ")
        env = make("connectx", debug=True)
        env.reset()
        # Play as the first agent against default "random" agent.
        env.run([agent, agent])
        output = env.render(mode="html", width=500, height=450)
        with open(file_name, "w") as f:
            f.write(output)
            f.flush()
        print("Rendered to {}".format(file_name))

    def extract_state(self, observation):
        state = []
        for cell in observation['board']:
            if cell == 0:
                state.append(0.0)
            elif cell == observation.mark:
                state.append(1.0)
            else:
                state.append(-1.0)
        return np.array(state).reshape(
            (1, self.env.configuration.rows, self.env.configuration.columns, 1))
