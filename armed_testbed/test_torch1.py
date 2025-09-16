import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import torch

def estimate(num_episodes=1000, num_env=100, k=10, epsilon=0.1, initial=0):
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    average_rewards, average_best_options = estimate(num_episodes=num_episodes, num_env=num_env, k=10, epsilon=0.1, initial=0)
    armed.drawAverageRewards(average_rewards.cpu().numpy(), num_episodes, ['epsilon=0.1'], xlim=(0,1000), ylim=(0,1.5), save_path='average_reward_test_torch1.png')
    armed.drawAverageBestOption(average_best_options.cpu().numpy()*100, num_episodes, ['epsilon=0.1'], xlim=(0,1000), ylim=(0,100), save_path='average_best_option_test_torch1.png')

if __name__ == "__main__":
    main()