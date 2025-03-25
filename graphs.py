import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


COLORS = ['#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        '#aec7e8',
        '#ffbb78',
        '#98df8a',
        '#ff9896',]


def plot_preferential_speed(preferential_speed, ax):
    preferential_speed_plotted = []
    for i_subject, subject in enumerate(preferential_speed.keys()):
        nb = 0
        if preferential_speed[subject] in preferential_speed_plotted:
            nb = preferential_speed_plotted.count(preferential_speed[subject])
        ax.vlines(preferential_speed[subject], 0 + 1.8*nb, 1.5 + 1.8*nb, linestyles="-", color=COLORS[i_subject])
        ax.plot(preferential_speed[subject], 0.25 + 1.8*nb, "v", color=COLORS[i_subject])
        preferential_speed_plotted += [preferential_speed[subject]]
    ax.set_ylabel("Preferential speed")
    ax.set_ylim(0, 10)
    ax.set_xlim(0.4, 1.6)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)


def plot_energenic_data(cw, cw_opt, preferential_speed):

    fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]})

    plot_preferential_speed(preferential_speed, axs[0, 0])

    tested_speeds = np.array([0.5, 0.75, 1, 1.25, 1.5])
    for i_subject, subject in enumerate(cw.keys()):
        axs[0, 1].plot(tested_speeds, cw[subject], ".", linestyle="-", linewidth=0.5, color=COLORS[i_subject])
        # ax.plot(cw_opt[subject][0], cw_opt[subject][1], "x", color=COLORS[i_subject])
        axs[0, 1].vlines(cw_opt[subject][0], 0, 10, linestyles="-", color=COLORS[i_subject])
    axs[0, 1].set_ylabel("Energy spent [units]")
    axs[0, 1].set_xlim(0.4, 1.6)
    axs[0, 1].set_ylim(2.15, 8.15)
    axs[0, 1].axes.get_xaxis().set_visible(False)

    for i_subject, subject in enumerate(cw.keys()):
        diff = cw_opt[subject][0] - preferential_speed[subject]
        if diff > 0:
            axs[1, 1].broken_barh([(0, diff)], (-i_subject*0.1, -(i_subject+1)*0.1), facecolors=COLORS[i_subject])
    axs[1, 1].vlines(0, 0, 10, linestyles="-", color="black")

    # EMG, EMG_opt

    fig.subplots_adjust(hspace=0.05, top=1.0)
    fig.savefig("energetic_data.png")
    plt.show()


def plot_variability(lyapunov_exponent, std_angles):

    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1]})

    # Lyapunov

    tested_speeds = np.array([0.75, 1.00, 1.25])
    for i_subject, subject in enumerate(std_angles.keys()):
        sorded_velocities_idx = np.argsort([float(key) for key in std_angles[subject].keys()])
        sorted_keys = np.array(list(std_angles[subject].keys()))[sorded_velocities_idx]
        sorted_velocities = np.array([float(key) for key in sorted_keys])
        std_array = np.array([std_angles[subject][key] for key in sorted_keys])
        axs[0, 1].plot(tested_speeds, std_array, ".", linestyle="-", linewidth=0.5, color=COLORS[i_subject])
        # TODO: see if I want to do a quadratic fit
        # axs[0, 1].vlines(cw_opt[subject][0], 0, 10, linestyles="-", color=COLORS[i_subject])
    axs[0, 1].set_ylabel("Energy spent [units]")
    axs[0, 1].set_xlim(0.4, 1.6)
    # axs[0, 1].set_ylim(2.15, 8.15)
    axs[0, 1].axes.get_xaxis().set_visible(False)


    # for i_subject, subject in enumerate(cw.keys()):
    #     diff = cw_opt[subject][0] - preferential_speed[subject]
    #     if diff > 0:
    #         axs[1, 1].broken_barh([(0, diff)], (-i_subject*0.1, -(i_subject+1)*0.1), facecolors=COLORS[i_subject])
    # axs[1, 1].vlines(0, 0, 10, linestyles="-", color="black")

    # EMG, EMG_opt

    fig.subplots_adjust(hspace=0.05, top=1.0)
    fig.savefig("energetic_data.png")
    plt.show()

def plot_stability(h_5_to_59_percentile, mean_com_acceleration):



def plot_data(data):

    cw = data["cw"]
    cw_opt = data["cw_opt"]
    preferential_speed = data["preferential_speed"]
    lyapunov_exponent = data["lyapunov_exponent"]
    std_angles = data["std_angles"]
    h_5_to_59_percentile = data["h_5_to_59_percentile"]
    mean_com_acceleration = data["mean_com_acceleration"]


    plot_energenic_data(cw, cw_opt, preferential_speed)

    plot_variability(lyapunov_exponent, std_angles)

    plot_stability(h_5_to_59_percentile, mean_com_acceleration)





















