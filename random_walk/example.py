from pkg import simulate, monte_carlo_estimate, dt_estimate, batch
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    avr_times = 100
    sim_times = 100
    mc_alpha_linestyle = [
        {"alpha": 0.005, "linestyle": "-"},
        {"alpha": 0.02, "linestyle": "--"},
        # {"alpha": 0.03, "linestyle": ":"}
    ]
    dc_alpha_linestyle = [
        {"alpha": 0.01, "linestyle": "-"},
        {"alpha": 0.1, "linestyle": "--"},
        # {"alpha": 0.15, "linestyle": ":"}
    ]
    mc_alpha = [item["alpha"] for item in mc_alpha_linestyle]
    dc_alpha = [item["alpha"] for item in dc_alpha_linestyle]

    nums = 2 * (len(mc_alpha) + len(dc_alpha))

    def batch_mc(episodes, alpha):
        yield from monte_carlo_estimate(batch(episodes), alpha=alpha)
    def batch_dt(episodes, alpha):
        yield from dt_estimate(batch(episodes), alpha=alpha)
    def mc(episodes, alpha):
        yield from monte_carlo_estimate(episodes, alpha=alpha)
    def dt(episodes, alpha):
        yield from dt_estimate(episodes, alpha=alpha)


    all_thing = [
        {"label": f"MC alpha={item['alpha']}", "func": mc, "alpha": item["alpha"], "color": "red", "linestyle": item["linestyle"], "data": np.zeros(sim_times+1)} for item in mc_alpha_linestyle
    ] + [
        {"label": f"DC alpha={item['alpha']}", "func": dt, "alpha": item["alpha"], "color": "blue", "linestyle": item["linestyle"], "data": np.zeros(sim_times+1)} for item in dc_alpha_linestyle
    ] + [
        {"label": f"Batch MC alpha={item['alpha']}", "func": batch_mc, "alpha": item["alpha"], "color": "yellow", "linestyle": item["linestyle"], "data": np.zeros(sim_times+1)} for item in mc_alpha_linestyle
    ] + [
        {"label": f"Batch DC alpha={item['alpha']}", "func": batch_dt, "alpha": item["alpha"], "color": "cyan", "linestyle": item["linestyle"], "data": np.zeros(sim_times+1)} for item in dc_alpha_linestyle
    ]

    for _ in range(avr_times):
        for item in all_thing:
            rms = [iter["RMS"] for iter in item["func"](simulate(sim_times=sim_times), alpha=item["alpha"])]
            item["data"] += np.array(rms)
    for item in all_thing:
        item["data"] = item["data"] / avr_times
        plt.plot(item["data"], label=item["label"], color=item["color"], linestyle=item["linestyle"])

    plt.xlabel("Episodes")
    plt.ylabel("RMS Error")
    plt.ylim(0, 0.253)
    plt.legend()
    plt.savefig(os.path.dirname(os.path.abspath(__file__))+'/plots/runs.png')
    plt.close()


