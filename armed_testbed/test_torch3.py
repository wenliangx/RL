import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import torch

def estimate(num_episodes=1000, num_env=100, k=10, alpha=0.1, initial=0, device='cuda'):
    env = armed.BaseEnvironmentTorch(k=k, num_env=num_env, device=device)
    agent = armed.GradientAgentTorch(k=k, num_env=num_env, initial=initial, alpha=alpha, device=device)

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

    labels = ['alpha=0.1', 'alpha=0.4']
    alpha_values = [0.1, 0.4]
    average_rewards = np.zeros((num_episodes, len(labels)))
    average_best_options = np.zeros((num_episodes, len(labels)))
    for i, alpha in enumerate(alpha_values):
        average_reward, average_best_option = estimate(num_episodes=num_episodes, num_env=num_env, k=k, alpha=alpha, initial=0, device=device)
        average_rewards[:, i] = average_reward.cpu().numpy()
        average_best_options[:, i] = average_best_option.cpu().numpy()
    armed.drawAverageRewards(average_rewards, num_episodes, labels, xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test_torch3.png')
    armed.drawAverageBestOption(average_best_options * 100, num_episodes, labels, xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test_torch3.png')

if __name__ == "__main__":
    main()