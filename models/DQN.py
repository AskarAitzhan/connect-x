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

        conv_vertical_a = Conv2D(filters=32, kernel_size=(self.inarow, 1), padding='same', activation='swish')(input_layer)
        conv_vertical_b = Conv2D(filters=64, kernel_size=(self.rows, 1), activation='swish')(conv_vertical_a)
        conv_vertical_c = Flatten()(conv_vertical_b)

        conv_horizontal_a = Conv2D(filters=32, kernel_size=(1, self.inarow), padding='same', activation='swish')(input_layer)
        conv_horizontal_b = Conv2D(filters=64, kernel_size=(self.rows, 1), activation='swish')(conv_horizontal_a)
        conv_horizontal_c = Flatten()(conv_horizontal_b)

        assert self.inarow % 2 == 0, "inarow must be even, for square the convolution layer to work"
        half_inarow = int(self.inarow / 2)
        conv_square_a = Conv2D(filters=32, kernel_size=(half_inarow, half_inarow), padding='same', activation='swish')(input_layer)
        conv_square_b = Conv2D(filters=32, kernel_size=(half_inarow, half_inarow), padding='same', activation='swish')(conv_square_a)
        conv_square_c = Conv2D(filters=128, kernel_size=(self.rows, 1), activation='swish')(conv_square_b)
        conv_square_d = Flatten()(conv_square_c)

        merged = Concatenate()([conv_vertical_c, conv_horizontal_c, conv_square_d])

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
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice(np.arange(configuration.columns)))

        act_values = self.target(self.extract_state(observation))[0]
        probs = softmax(act_values)
        chosen_index = np.random.choice(np.arange(len(act_values)), p=probs)
        return int(chosen_index)

    def agent(self, observation, configuration):
        act_values = self.target(self.extract_state(observation))[0]
        for i, val in enumerate(act_values):
            if observation['board'][i] != 0:
                act_values[i] = -1000
        return int(np.argmax(act_values))

    def train(self, episodes):
        for episode in range(episodes):
            self.init_episode(episode)
            self.run_episode()

    def init_episode(self, episode):
        print("Initializing episode #{} ...".format(episode))
        if episode % self.render_frequency == 0:
            self.render(self.agent, "dqn{}.html".format(episode))
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
