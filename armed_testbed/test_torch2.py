import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import torch

def estimate(num_episodes=1000, num_env=100, k=10, c=2, initial=0):
    env = armed.BaseEnvironmentTorch(k=k, num_env=num_env)
    agent = armed.UpperConfidenceBoundAgentTorch(k=k, num_env=num_env, initial=initial, c=c)

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

def estimateEpsilon(num_episodes=1000, num_env=100, k=10, epsilon=0.1, initial=0):
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
    num_episodes = 1000
    num_env = 2000
    k = 10 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    labels = ['c=2', 'epsilon=0.1']
    c = 2
    epsilon_values = 0.1
    average_rewards = np.zeros((num_episodes, len(labels)))
    average_best_options = np.zeros((num_episodes, len(labels)))

    average_reward, average_best_option = estimate(num_episodes=num_episodes, num_env=num_env, k=k, c=c, initial=0)
    average_rewards[:, 0] = average_reward.cpu().numpy()
    average_best_options[:, 0] = average_best_option.cpu().numpy()
    average_reward, average_best_option = estimateEpsilon(num_episodes=num_episodes, num_env=num_env, k=k, epsilon=epsilon_values, initial=0)
    average_rewards[:, 1] = average_reward.cpu().numpy()
    average_best_options[:, 1] = average_best_option.cpu().numpy()

    armed.drawAverageRewards(average_rewards, num_episodes, labels, xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test_torch2.png')
    armed.drawAverageBestOption(average_best_options * 100, num_episodes, labels, xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test_torch2.png')

if __name__ == "__main__":
    main()