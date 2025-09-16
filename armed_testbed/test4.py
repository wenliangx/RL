import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np


def gradientAgentEstimate(num_episodes=1000, k=10, alpha=0.1, baseline=True):
    env = armed.BaseEnvironment(k=k)
    agent = armed.GradientAgent(k=k, alpha=alpha, baseline=baseline)

    rewards = np.zeros(num_episodes)
    best_option = np.zeros(num_episodes)

    for episode in range(num_episodes):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward, alpha=0.0)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option

def main():
    test_num = 2000
    num_episodes = 1000
    k = 10
    alpha = [0.1, 0.4]
    all_rewards = np.zeros((test_num, num_episodes, len(alpha)))
    all_best_options = np.zeros((test_num, num_episodes, len(alpha)))

    for test in range(test_num):
        print(f'Test {test+1}/{test_num}')
        all_rewards[test, :, 0], all_best_options[test, :, 0] = gradientAgentEstimate(num_episodes=num_episodes, k=k, alpha=alpha[0], baseline=False)
        all_rewards[test, :, 1], all_best_options[test, :, 1] = gradientAgentEstimate(num_episodes=num_episodes, k=k, alpha=alpha[1], baseline=False)

    average_rewards = np.mean(all_rewards, axis=0)
    armed.drawAverageRewards(average_rewards, num_episodes, [f'alpha={alpha[0]}', f'alpha={alpha[1]}'], xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test4.png')

    average_best_options = np.mean(all_best_options, axis=0) * 100
    armed.drawAverageBestOption(average_best_options, num_episodes, [f'alpha={alpha[0]}', f'alpha={alpha[1]}'], xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test4.png')


if __name__ == '__main__':
    main()