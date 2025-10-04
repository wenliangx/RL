from pkg import Env, Policy
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    num_episode = 170
    num_experiments = 100
    data_x = np.zeros(num_episode)
    data_y = np.zeros(num_episode)

    for _ in range(num_experiments):
        times = 0
        data=[[],[]]
        policy = Policy(epsilon=0.9, alpha=0.5)
        env = Env()
        for i in range(num_episode):
            env.reset()
            for _ in env.run(policy):
                times += 1
            data[0].append(times)
            data[1].append(i)
        data_x += np.array(data[0])
        data_y += np.array(data[1])


    data_x /= num_experiments
    data_y /= num_experiments

    plt.plot(data_x, data_y)
    plt.show()
    plt.close()