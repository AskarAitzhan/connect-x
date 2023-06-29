from kaggle_environments import evaluate, make, utils

from agents.brute_force_agent import brute_force_agent
from agents.deep_q_network_agent import encode_weights, deep_q_network_agent
from models.DQN import DQN
import tensorflow as tf


def dumb_agent(observation, configuration):
    return 0

def render(agent, file_name="render.html"):
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


def train_dqn():
    dqn = DQN(make("connectx", debug=True), epsilon=1.0,
              epsilon_decay=0.998, tau=0.005, render_frequency=100, save_frequency=100)
    dqn.load("dqn-main.h5", "dqn-target.h5")
    dqn.train(10000)
    dqn.save("dqn-main.h5", "dqn-target.h5")


def main():
    render(deep_q_network_agent)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    main()
