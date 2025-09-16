import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
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

def estimateTorch(episode_num=1000, env_num=100, k=10, epsilon=0.1, initial=0):
    env = armed.BaseEnvironmentTorch(k=k, env_num=env_num)
    agent = armed.EpsilonAgentTorch(k=k, env_num=env_num, epsilon=epsilon, initial=initial)

    rewards = torch.zeros(episode_num, device=env.device)
    best_option = torch.zeros(episode_num, device=env.device)

    for episode in range(episode_num):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward)
        rewards[episode] = reward.mean().item()
        best_action = torch.argmax(env.q_star, dim=1)
        best_option[episode] = (action == best_action).float().mean().item()

    return rewards, best_option

def run(env_num, episode_num, k, epsilon_values):
    all_rewards = np.zeros((env_num, episode_num, len(epsilon_values)))
    all_best_options = np.zeros((env_num, episode_num, len(epsilon_values)))

    for test in range(env_num):
        print(f'Test {test+1}/{env_num}')
        for i, epsilon in enumerate(epsilon_values):
            all_rewards[test, :, i], all_best_options[test, :, i] = estimate(num_episodes=episode_num, k=k, epsilon=epsilon, initial=0)

    average_rewards = np.mean(all_rewards, axis=0)
    average_best_options = np.mean(all_best_options, axis=0)
    return average_rewards, average_best_options


def runTorch(env_num, episode_num, k, epsilon_values):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    average_rewards = np.zeros((episode_num, len(epsilon_values)))
    average_best_options = np.zeros((episode_num, len(epsilon_values)))
    for i, epsilon in enumerate(epsilon_values):
        average_reward, average_best_option = estimateTorch(episode_num=episode_num, env_num=env_num, k=k, epsilon=epsilon, initial=0)
        average_rewards[:, i] = average_reward.cpu().numpy()
        average_best_options[:, i] = average_best_option.cpu().numpy()
    return average_rewards, average_best_options
    
def draw(average_rewards, average_best_options, epsilon_values, colors):
    labels = [f"epsilon={epsilon}" for epsilon in epsilon_values]
    alpha_values = [0.8 for _ in epsilon_values]
    linestyles = ['-' for _ in epsilon_values]
    xlim = (0, 1000)
    ylim_reward = (0, 1.5)
    ylim_best_option = (0, 100)

    reward_plt_params = {
        'labels': labels,
        'colors': colors,
        'alpha_values': alpha_values,
        'linestyles': linestyles,
        'xlim': xlim,
        'ylim': ylim_reward,
        'save_path': 'average_reward1.png',
        'title': 'Average Reward over Time',
        'xlabel': 'Episodes',
        'ylabel': 'Average Reward'
    }
    best_option_plt_params = {
        'labels': labels,
        'colors': colors,
        'alpha_values': alpha_values,
        'linestyles': linestyles,
        'xlim': xlim,
        'ylim': ylim_best_option,
        'save_path': 'average_best_option1.png',
        'title': 'Average Best Option over Time',
        'xlabel': 'Episodes',
        'ylabel': 'Average Best Option (%)'
    }
    armed.drawAverageRewards(rewards=average_rewards, 
                             **reward_plt_params)
    armed.drawAverageBestOption(best_options=average_best_options * 100, 
                                **best_option_plt_params)
if __name__ == "__main__":
    env_num = 2000
    episode_num = 1000
    k = 10
    epsilon_values = [0.1, 0.01, 0.0]
    colors = ['blue', 'red', 'green']
    average_rewards, average_best_options = runTorch(env_num=env_num, episode_num=episode_num, k=k, epsilon_values=epsilon_values)
    # average_rewards, average_best_options = run(env_num=env_num, episode_num=episode_num, k=k, epsilon_values=epsilon_values)
    draw(average_rewards, average_best_options, epsilon_values=epsilon_values, colors=colors)