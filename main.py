from kaggle_environments import evaluate, make, utils

from agents.brute_force_agent import brute_force_agent

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = make("connectx", debug=True)
    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([brute_force_agent, brute_force_agent])
    output = env.render(mode="html", width=500, height=450)
    with open("render.html", "w") as f:
        f.write(output)
