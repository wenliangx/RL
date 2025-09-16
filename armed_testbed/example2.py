import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import torch


def epsilonAgentEstimate(episode_num=1000, k=10, epsilon=0.1, initial=0):
    env = armed.BaseEnvironment(k=k)
    agent = armed.EpsilonAgent(k=k, epsilon=epsilon, initial=initial)

    rewards = np.zeros(episode_num)
    best_option = np.zeros(episode_num)

    for episode in range(episode_num):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward, alpha=0.1)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option

def upperConfidenceBoundAgentEstimate(episode_num=1000, k=10, c=2, initial=0):
    env = armed.BaseEnvironment(k=k)
    agent = armed.UpperConfidenceBoundAgent(k=k, initial=initial)

    rewards = np.zeros(episode_num)
    best_option = np.zeros(episode_num)

    for episode in range(episode_num):
        action = agent.select_action(c=c)
        reward = env.step(action)
        agent.update_estimation(action, reward)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option

def upperConfidenceBoundAgentEstimateTorch(episode_num=1000, env_num=100, k=10, c=2, initial=0):
    env = armed.BaseEnvironmentTorch(k=k, env_num=env_num)
    agent = armed.UpperConfidenceBoundAgentTorch(k=k, env_num=env_num, initial=initial, c=c)

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

def epsilonAgentEstimateTorch(episode_num=1000, env_num=100, k=10, epsilon=0.1, initial=0):
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


def run(env_num, episode_num, k, c, epsilon):

    all_rewards = np.zeros((env_num, episode_num, 2))
    all_best_options = np.zeros((env_num, episode_num, 2))

    for test in range(env_num):
        print(f'Test {test+1}/{env_num}')
        all_rewards[test, :, 0], all_best_options[test, :, 0] = upperConfidenceBoundAgentEstimate(episode_num=episode_num, k=k, c=c, initial=0)
        all_rewards[test, :, 1], all_best_options[test, :, 1] = epsilonAgentEstimate(episode_num=episode_num, k=k, epsilon=epsilon, initial=0)

    average_rewards = np.mean(all_rewards, axis=0)
    average_best_options = np.mean(all_best_options, axis=0)

    return average_rewards, average_best_options
    


def runTorch(env_num, episode_num, k, c, epsilon):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    average_rewards = np.zeros((episode_num, 2))
    average_best_options = np.zeros((episode_num, 2))

    average_reward, average_best_option = upperConfidenceBoundAgentEstimateTorch(episode_num=episode_num, env_num=env_num, k=k, c=c, initial=0)
    average_rewards[:, 0] = average_reward.cpu().numpy()
    average_best_options[:, 0] = average_best_option.cpu().numpy()
    average_reward, average_best_option = epsilonAgentEstimateTorch(episode_num=episode_num, env_num=env_num, k=k, epsilon=epsilon, initial=0)
    average_rewards[:, 1] = average_reward.cpu().numpy()
    average_best_options[:, 1] = average_best_option.cpu().numpy()

    return average_rewards, average_best_options




def draw(average_rewards, average_best_options, c, epsilon, colors, alpha_values):
    labels = [f"c={c}", f"epsilon={epsilon}"]
    linestyles = ['-', '-']

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
        'save_path': 'average_reward2.png',
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
        'save_path': 'average_best_option2.png',
        'title': 'Average Best Option over Time',
        'xlabel': 'Episodes',
        'ylabel': 'Average Best Option (%)'
    }

    armed.drawAverageRewards(rewards=average_rewards, 
                             **reward_plt_params)
    armed.drawAverageBestOption(best_options=average_best_options * 100, 
                                **best_option_plt_params)

if __name__ == '__main__':
    env_num = 2000
    episode_num = 1000
    k = 10 
    c = 2
    epsilon = 0.1
    colors = ['blue', 'gray']
    alpha_values = [0.8, 0.3]
    average_rewards, average_best_options = runTorch(env_num=env_num, episode_num=episode_num, k=k, c=c, epsilon=epsilon)
    # average_rewards, average_best_options = run(env_num=env_num, episode_num=episode_num, k=k, c=c, epsilon=epsilon)
    draw(average_rewards, average_best_options, c=c, epsilon=epsilon, colors=colors, alpha_values=alpha_values)