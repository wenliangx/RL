import numpy as np 
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, k=10, mean=0, std=1):
        self.k = k
        self.mean = mean
        self.std = std
        self.q_star = np.random.normal(self.mean, self.std, self.k)

    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1)
        return reward


class Agent:
    def __init__(self, k=10, epsilon=0.1, initial=0):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_estimation)

    def update_estimation(self, action, reward):
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

def draw_average_rewards(rewards: np.ndarray, num_episodes: int, labels: list, xlim=None, ylim=None):
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

def draw_average_best_option(best_options: np.ndarray, num_episodes: int, labels: list, xlim=None, ylim=None):
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

def estimate(num_episodes=1000, k=10, epsilon=0.1, initial=0):
    env = Environment(k=k)
    agent = Agent(k=k, epsilon=epsilon, initial=initial)

    rewards = np.zeros(num_episodes)
    best_option = np.zeros(num_episodes)

    for episode in range(num_episodes):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option


def main():
    test_num = 2000
    num_episodes = 1000
    k = 10
    epsilons = [0.1, 0.01, 0.0]
    all_rewards = np.zeros((test_num, num_episodes, len(epsilons)))
    all_best_options = np.zeros((test_num, num_episodes, len(epsilons)))

    for test in range(test_num):
        print(f'Test {test+1}/{test_num}')
        for i, epsilon in enumerate(epsilons):
            all_rewards[test, :, i], all_best_options[test, :, i] = estimate(num_episodes=num_episodes, k=k, epsilon=epsilon, initial=0)

    average_rewards = np.mean(all_rewards, axis=0)
    draw_average_rewards(average_rewards, num_episodes, ['epsilon=0.1', 'epsilon=0.01', 'epsilon=0.0'], xlim=(0,1000), ylim=(0,1.5))

    average_best_options = np.mean(all_best_options, axis=0) * 100
    draw_average_best_option(average_best_options, num_episodes, ['epsilon=0.1', 'epsilon=0.01', 'epsilon=0.0'], xlim=(0,1000), ylim=(0,100))

if __name__ == "__main__":
    main()