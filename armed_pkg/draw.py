import numpy as np
import matplotlib.pyplot as plt

def drawAverageRewards(rewards: np.ndarray, num_episodes: int, labels: list, xlim=None, ylim=None):
    if rewards.shape[1] != len(labels):
        raise ValueError("Input lists must have the same length.")
    for i, label in enumerate(labels):
        plt.plot((np.arange(num_episodes) + 1), rewards[:, i], label=label)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()  # 添加这一行显示label
    plt.savefig('average_reward.png')
    plt.close()

def drawAverageBestOption(best_options: np.ndarray, num_episodes: int, labels: list, xlim=None, ylim=None):
    if best_options.shape[1] != len(labels):
        raise ValueError("Input lists must have the same length.")
    for i, label in enumerate(labels):
        plt.plot((np.arange(num_episodes) + 1), best_options[:, i], label=label)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('Episodes')
    plt.ylabel('Average Best Option')
    plt.title('Average Best Option over Time')
    plt.legend()  # 添加这一行显示label
    plt.savefig('average_best_option.png')
    plt.close()