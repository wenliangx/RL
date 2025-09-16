import numpy as np
import matplotlib.pyplot as plt

def __decode_kwargs(data_len, **kwargs):
    labels = kwargs.get('labels', None)
    if labels is None:
        Warning("No labels provided, defaulting to generic labels.")
        labels = [f'Line {i+1}' for i in range(data_len)]
    elif len(labels) != data_len:
        raise ValueError("Length of labels must match number of columns in rewards.")

    colors = kwargs.get('colors', None)
    if colors is not None and len(colors) != data_len:
        raise ValueError("Length of colors must match number of columns in rewards.")
    
    alpha_values = kwargs.get('alpha_values', 0.7*np.ones(data_len))

    linestyles = kwargs.get('linestyles', ['-'] * data_len)
    if len(linestyles) != data_len:
        raise ValueError("Length of linestyles must match number of columns in rewards.")
    
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    save_path = kwargs.get('save_path', None)
    title = kwargs.get('title', None)
    xlabel = kwargs.get('xlabel', 'Episodes')
    ylabel = kwargs.get('ylabel', None)
    return labels, colors, alpha_values, linestyles, xlim, ylim, save_path, title, xlabel, ylabel

def __draw(datas: np.ndarray, default_ylabel: str, default_title: str, default_save_path: str, **kwargs):
    if np.ndim(datas) == 1:
        datas = datas.reshape(-1, 1)

    labels, colors, alpha_values, linestyles, xlim, ylim, save_path, title, xlabel, ylabel = __decode_kwargs(datas.shape[1], **kwargs)
    episode_num = datas.shape[0]
    for i in range(datas.shape[1]):
        if colors:
            plt.plot((np.arange(episode_num) + 1), datas[:, i], label=labels[i], color=colors[i], alpha=alpha_values[i], linestyle=linestyles[i])
        else:
            plt.plot((np.arange(episode_num) + 1), datas[:, i], label=labels[i], alpha=alpha_values[i], linestyle=linestyles[i])
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else default_ylabel)
    plt.title(title if title else default_title)
    plt.legend()
    plt.savefig(save_path if save_path else default_save_path)
    plt.close()

def drawAverageRewards(rewards: np.ndarray, **kwargs):
    __draw(datas=rewards, 
           default_ylabel='Average Reward', 
           default_title='Average Reward over Time', 
           default_save_path='average_rewards.png', **kwargs)

def drawAverageBestOption(best_options: np.ndarray, **kwargs):
    __draw(datas=best_options,
           default_ylabel='Average Best Option', 
           default_title='Average Best Option over Time', 
           default_save_path='average_best_option.png', **kwargs)