import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import time
import torch

def estimate(num_episodes=1000, k=10, epsilon=0.1, initial=0):
    env = armed.BaseEnvironment(k=k)
    agent = armed.EpsilonAgent(k=k, epsilon=epsilon, initial=initial)

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

def estimateTorch(num_episodes=1000, num_env=100, k=10, epsilon=0.1, initial=0):
    env = armed.BaseEnvironmentTorch(k=k, num_env=num_env)
    agent = armed.EpsilonAgentTorch(k=k, num_env=num_env, epsilon=epsilon, initial=initial)

    rewards = torch.zeros(num_episodes, device=env.device)
    best_option = torch.zeros(num_episodes, device=env.device)

    for episode in range(num_episodes):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward)
        rewards[episode] = reward.mean().item()
        best_action = torch.argmax(env.q_star, dim=1)
        best_option[episode] = (action == best_action).float().mean().item()

    return rewards, best_option


def main():
    test_num = 2000
    num_episodes = 1000
    k = 10
    epsilons = [0.1, 0.01, 0.0]
    labels = ['epsilon=0.1', 'epsilon=0.01', 'epsilon=0.0']
    all_rewards = np.zeros((test_num, num_episodes, len(epsilons)))
    all_best_options = np.zeros((test_num, num_episodes, len(epsilons)))

    for test in range(test_num):
        print(f'Test {test+1}/{test_num}')
        for i, epsilon in enumerate(epsilons):
            all_rewards[test, :, i], all_best_options[test, :, i] = estimate(num_episodes=num_episodes, k=k, epsilon=epsilon, initial=0)

    # average_rewards = np.mean(all_rewards, axis=0)
    # armed.drawAverageRewards(average_rewards, num_episodes, labels, xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test1.png')

    # average_best_options = np.mean(all_best_options, axis=0) * 100
    # armed.drawAverageBestOption(average_best_options, num_episodes, labels, xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test1.png')

def mainTorch():
    num_episodes = 1000
    num_env = 2000
    k = 10 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    labels = ['epsilon=0.1', 'epsilon=0.01', 'epsilon=0.0']
    epsilon_values = [0.1, 0.01, 0.0]
    average_rewards = np.zeros((num_episodes, len(labels)))
    average_best_options = np.zeros((num_episodes, len(labels)))
    for i, epsilon in enumerate(epsilon_values):
        average_reward, average_best_option = estimateTorch(num_episodes=num_episodes, num_env=num_env, k=k, epsilon=epsilon, initial=0)
        average_rewards[:, i] = average_reward.cpu().numpy()
        average_best_options[:, i] = average_best_option.cpu().numpy()
    # armed.drawAverageRewards(average_rewards, num_episodes, labels, xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test_torch1.png')
    # armed.drawAverageBestOption(average_best_options * 100, num_episodes, labels, xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test_torch1.png')



if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start} seconds")
    time_start = time.time()
    mainTorch()
    time_end = time.time()
    print(f"Total time taken (Torch): {time_end - time_start} seconds")