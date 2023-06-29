from kaggle_environments import evaluate, make, utils

from agents import combined, combined_during_training
from agents.brute_force_agent import brute_force_agent
from agents.combined import combined_after_training_agent
from agents.combined_during_training import combined_during_training_agent
from agents.deep_q_network_agent import encode_weights, deep_q_network_agent
from models.DQN import DQN
import tensorflow as tf


def dumb_agent(observation, configuration):
    return 0

def render(agent1, agent2=None, file_name="render.html"):
    print("Rendering ... ")
    env = make("connectx", debug=True)
    env.reset()
    # Play as the first agent against default "random" agent.
    reward = [0, 0]
    logs = env.run([agent1, agent1 if agent2 is None else agent2])
    for steps in logs:
        for turn in steps:
            reward[turn['observation']['mark'] - 1] += turn['reward'] if turn['reward'] is not None else 0
    print("Total reward: {}".format(reward))
    output = env.render(mode="html", width=500, height=450)
    with open(file_name, "w") as f:
        f.write(output)
        f.flush()
    print("Rendered to {}".format(file_name))
    return reward

def train_dqn():
    dqn = DQN(make("connectx", debug=True), epsilon=1.0,
              epsilon_decay=0.995, tau=0.005, render_frequency=10, save_frequency=10)
    dqn.load("dqn-main.h5", "dqn-target.h5")
    dqn.train(10000)
    dqn.save("dqn-main.h5", "dqn-target.h5")


def evaluate_agent(subject):
    total_games = 2

    opponents = ["random", "negamax", brute_force_agent, deep_q_network_agent, combined_after_training_agent, combined_during_training_agent]
    opponents_name = ["random", "negamax", "brute_force_agent", "deep_q_network_agent", "combined_after_training_agent", "combined_during_training_agent"]
    for opponent, name in zip(opponents, opponents_name):
        total_wins = 0
        total_losses = 0
        for i in range(total_games):
            if i % 2 == 0:
                res = render(subject, opponent)
                total_wins += max(res[0], 0)
                total_losses += max(res[1], 0)
            else:
                res = render(opponent, subject)
                total_wins += max(res[1], 0)
                total_losses += max(res[0], 0)
        total_draws = total_games - total_wins - total_losses
        print("Total wins against {} opponent: {} out of {} games. Win percentage: {}%".format(name, total_wins, total_games, int((total_wins / total_games) * 100)))
        print("Total losses against {} opponent: {} out of {} games. Loss percentage: {}%".format(name, total_losses, total_games, int((total_losses / total_games) * 100)))
        print("Total draws against {} opponent: {} out of {} games. Draw percentage: {}%".format(name, total_draws, total_games, int((total_draws / total_games) * 100)))



def main():
    evaluate_agent(combined_during_training_agent)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    main()
