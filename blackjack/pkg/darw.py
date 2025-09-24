import matplotlib.pyplot as plt
import numpy as np

def plot_blackjack(ax, V):
    ax.view_init(azim=-150)
    Y = np.arange(1,11)
    X = np.arange(11,21)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, V, cmap='summer')