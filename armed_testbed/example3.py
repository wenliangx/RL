import sys
sys.path.append('/ws')
import armed_pkg as armed
import numpy as np
import torch

def estimate(episode_num=1000, k=10, alpha=0.1, baseline=True):
    env = armed.BaseEnvironment(k=k, mean=4)
    agent = armed.GradientAgent(k=k, alpha=alpha, baseline=baseline)

    rewards = np.zeros(episode_num)
    best_option = np.zeros(episode_num)

    for episode in range(episode_num):
        action = agent.select_action()
        reward = env.step(action)
        agent.update_estimation(action, reward)
        rewards[episode] = reward
        if action == np.argmax(env.q_star):
            best_option[episode] = 1

    return rewards, best_option


def estimateTorch(episode_num=1000, env_num=100, k=10, alpha=0.1, baseline=True, initial=0, device='cuda'):
    env = armed.BaseEnvironmentTorch(k=k, env_num=env_num, device=device, mean=4)
    agent = armed.GradientAgentTorch(k=k, env_num=env_num, initial=initial, alpha=alpha, baseline=baseline, device=device)

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

def run(env_num, episode_num, k, conditions):
    all_rewards = np.zeros((env_num, episode_num, len(conditions)))
    all_best_options = np.zeros((env_num, episode_num, len(conditions)))

    for env in range(env_num):
        print(f'Env {env+1}/{env_num}')
        for i, condition in enumerate(conditions):
            all_rewards[env, :, i], all_best_options[env, :, i] = estimate(episode_num=episode_num, k=k, alpha=condition['alpha'], baseline=condition['baseline'])

    average_rewards = np.mean(all_rewards, axis=0)
    average_best_options = np.mean(all_best_options, axis=0)
    return average_rewards, average_best_options



def runTorch(env_num, episode_num, k, conditions):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    average_rewards = np.zeros((episode_num, len(conditions)))
    average_best_options = np.zeros((episode_num, len(conditions)))

    for i, condition in enumerate(conditions):
        average_reward, average_best_option = estimateTorch(episode_num=episode_num, 
                                                       env_num=env_num, 
                                                       k=k, 
                                                       alpha=condition['alpha'], 
                                                       baseline=condition['baseline'], 
                                                       initial=0, 
                                                       device=device)
        average_rewards[:, i] = average_reward.cpu().numpy()
        average_best_options[:, i] = average_best_option.cpu().numpy()

    return average_rewards, average_best_options
    



def draw(average_rewards, average_best_options, conditions, colors, alpha_values):
    labels = [f"alpha={conditions[i]['alpha']}, baseline={conditions[i]['baseline']}" for i in range(len(conditions))]
    xlim = (0, 1000)
    # ylim_reward = (0, 1.5)
    ylim_best_option = (0, 100)
    linestyles = ['-', '-', '-', '-']

    reward_plt_params = {
        'labels': labels,
        'colors': colors,
        'alpha_values': alpha_values,
        'linestyles': linestyles,
        'xlim': xlim,
        # 'ylim': ylim_reward,
        'save_path': 'average_reward3.png',
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
        'save_path': 'average_best_option3.png',
        'title': 'Average Best Option over Time',
        'xlabel': 'Episodes',
        'ylabel': 'Average Best Option (%)'
    }

    armed.drawAverageRewards(rewards=average_rewards, 
                             **reward_plt_params)
    armed.drawAverageBestOption(average_best_options * 100, 
                                **best_option_plt_params)
if __name__ == '__main__':
    env_num = 2000
    episode_num = 1000
    k = 10 
    alpha_list = [0.1, 0.4]
    baseline_list = [True, False]
    conditions = [{'alpha': alpha, 'baseline': baseline} for baseline in baseline_list for alpha in alpha_list]
    print(conditions)
    colors = ['blue', 'blue', 'orange', 'orange']
    alpha_values = [0.8, 0.3, 0.8, 0.3]
    average_rewards, average_best_options = runTorch(env_num=env_num, episode_num=episode_num, k=k, conditions=conditions)
    # average_rewards, average_best_options = run(env_num=env_num, episode_num=episode_num, k=k, conditions=conditions)
    draw(average_rewards, average_best_options, conditions=conditions, colors=colors, alpha_values=alpha_values)