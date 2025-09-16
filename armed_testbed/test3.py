import sys
sys.path.append('..')
import armed_pkg as armed
import numpy as np


def epsilonAgentEstimate(num_episodes=1000, k=10, epsilon=0.1, initial=0):
    env = armed.BaseEnvironment(k=k)
    agent = armed.EpsilonAgent(k=k, epsilon=epsilon, initial=initial)

    rewards = np.zeros(num_episodes)
    best_option = np.zeros(num_episodes)

    for episode in range(num_episodes):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward, alpha=0.1)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option

def upperConfidenceBoundAgentEstimate(num_episodes=1000, k=10, c=2, initial=0):
    env = armed.BaseEnvironment(k=k)
    agent = armed.UpperConfidenceBoundAgent(k=k, initial=initial)

    rewards = np.zeros(num_episodes)
    best_option = np.zeros(num_episodes)

    for episode in range(num_episodes):
        action = agent.select_action(c=c)
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
    all_rewards = np.zeros((test_num, num_episodes, 2))
    all_best_options = np.zeros((test_num, num_episodes, 2))

    for test in range(test_num):
        print(f'Test {test+1}/{test_num}')
        all_rewards[test, :, 0], all_best_options[test, :, 0] = epsilonAgentEstimate(num_episodes=num_episodes, k=k, epsilon=0.1, initial=0)
        all_rewards[test, :, 1], all_best_options[test, :, 1] = upperConfidenceBoundAgentEstimate(num_episodes=num_episodes, k=k, c=2, initial=0)

    average_rewards = np.mean(all_rewards, axis=0)
    armed.drawAverageRewards(average_rewards, num_episodes, [f'epsilon=0.1', f'c={2}'], xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test3.png')

    average_best_options = np.mean(all_best_options, axis=0) * 100
    armed.drawAverageBestOption(average_best_options, num_episodes, [f'epsilon=0.1', f'c={2}'], xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test3.png')


if __name__ == '__main__':
    main()