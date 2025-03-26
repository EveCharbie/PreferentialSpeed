import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
]


def plot_data_points(data, ax, preferential_speed):

    opt_velocity, opt = {}, {}
    for i_subject, subject in enumerate(data.keys()):
        speed_list = [
            preferential_speed[subject] if key == "preferential_speed" else float(key) for key in data[subject].keys()
        ]
        sorded_velocities_idx = np.argsort(speed_list)
        sorted_keys = np.array(list(data[subject].keys()))[sorded_velocities_idx]
        sorted_velocities = np.array(speed_list)[sorded_velocities_idx]
        array = np.array([data[subject][key] for key in sorted_keys])
        ax.plot(sorted_velocities, array, ".", linestyle="-", linewidth=0.5, color=COLORS[i_subject])

        # Quadratic fit
        p = np.polyfit(sorted_velocities, array, 2)
        opt_velocity[subject] = -p[1] / (2 * p[0])
        opt[subject] = p[0] * (opt_velocity[subject]) ** 2 + p[1] * opt_velocity[subject] + p[2]
        ax.vlines(opt_velocity[subject], 0, 200, linestyles="-", color=COLORS[i_subject])
    ax.set_xlim(0.4, 1.6)
    ax.axes.get_xaxis().set_visible(False)
    return opt_velocity, opt


def plot_velocity_difference(data_opt, preferential_speed, ax):
    for i_subject, subject in enumerate(data_opt.keys()):
        diff = preferential_speed[subject] - data_opt[subject]
        rect = Rectangle((0, -0.1 * i_subject), diff, -0.1, facecolor=COLORS[i_subject])
        ax.add_patch(rect)
    ax.vlines(0, 0.1, -1.5, linestyles="-", color="black")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.5, 0.5)


def plot_preferential_speed(preferential_speed, ax):
    preferential_speed_plotted = []
    for i_subject, subject in enumerate(preferential_speed.keys()):
        nb = 0
        if preferential_speed[subject] in preferential_speed_plotted:
            nb = preferential_speed_plotted.count(preferential_speed[subject])
        ax.vlines(preferential_speed[subject], 0 + 1.8 * nb, 1.5 + 1.8 * nb, linestyles="-", color=COLORS[i_subject])
        ax.plot(preferential_speed[subject], 0.25 + 1.8 * nb, "v", color=COLORS[i_subject])
        preferential_speed_plotted += [preferential_speed[subject]]
    ax.set_ylabel("Preferential speed")
    ax.set_ylim(0, 10)
    ax.set_xlim(0.4, 1.6)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)


def plot_energenic_data(cw, cw_opt_velocity, mean_emg, preferential_speed):

    fig, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(10, 7))

    # Preferential speed arrows
    plot_preferential_speed(preferential_speed, axs[0, 0])
    axs[0, 1].axes.get_xaxis().set_visible(False)
    axs[0, 1].axes.get_yaxis().set_visible(False)
    for side in ["top", "right", "bottom", "left"]:
        axs[0, 1].spines[side].set_visible(False)

    # K5 energy consumption
    tested_speeds = np.array([0.5, 0.75, 1, 1.25, 1.5])
    for i_subject, subject in enumerate(cw.keys()):
        axs[1, 0].plot(tested_speeds, cw[subject], ".", linestyle="-", linewidth=0.5, color=COLORS[i_subject])
        # ax.plot(cw_opt[subject][0], cw_opt[subject][1], "x", color=COLORS[i_subject])
        axs[1, 0].vlines(cw_opt_velocity[subject], 0, 10, linestyles="-", color=COLORS[i_subject])
    axs[1, 0].set_ylabel("Energy spent [units]")
    axs[1, 0].set_xlim(0.4, 1.6)
    axs[1, 0].set_ylim(2.15, 8.15)
    axs[1, 0].axes.get_xaxis().set_visible(False)

    plot_velocity_difference(cw_opt_velocity, preferential_speed, axs[1, 1])

    # EMG
    emg_opt_velocity, emg_opt = plot_data_points(mean_emg, axs[1, 0], preferential_speed)
    axs[1, 0].set_ylabel("mean absolute EMG [V]")
    # axs[1, 0].set_ylim(15, 175)
    plot_velocity_difference(emg_opt_velocity, preferential_speed, axs[1, 1])

    fig.subplots_adjust(hspace=0.05, top=1.0)
    fig.savefig("energetic_data.png")
    plt.show()


def plot_variability(lyapunov_exponent, std_angles, preferential_speed):

    fig, axs = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(10, 7))

    # STD kinematics
    std_opt_velocity, std_opt = plot_data_points(std_angles, axs[0, 0], preferential_speed)
    axs[0, 0].set_ylabel("Joint angles standard\ndeviation [ ]")
    axs[0, 0].set_ylim(15, 175)
    plot_velocity_difference(std_opt_velocity, preferential_speed, axs[0, 1])

    # Lyapunov
    lyap_opt_velocity, lyap_opt = plot_data_points(lyapunov_exponent, axs[1, 0], preferential_speed)
    axs[1, 0].set_ylabel("Largest Lyapunov exponent\nof joint angles [ ]")
    # axs[0, 0].set_ylim(15, 175)
    plot_velocity_difference(lyap_opt_velocity, preferential_speed, axs[1, 1])

    fig.subplots_adjust(hspace=0.05)
    fig.savefig("variability_data.png")
    plt.show()


def plot_stability(h_5_to_59_percentile, mean_com_acceleration, preferential_speed):

    fig, axs = plt.subplots(2, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(10, 7))

    # CoM acceleration
    com_opt_velocity, com_opt = plot_data_points(mean_com_acceleration, axs[0, 0], preferential_speed)
    axs[0, 0].set_ylabel(r"Center of mass acceleration\n[$m/s^2$]")
    axs[0, 0].set_ylim(0, 4)

    plot_velocity_difference(com_opt_velocity, preferential_speed, axs[0, 1])

    # Angular momentum
    h_opt_velocity, h_opt = plot_data_points(h_5_to_59_percentile, axs[0, 0], preferential_speed)
    axs[1, 0].set_ylabel(r"Angular momentum 5-95\npercetile [$kg m^2 rad/s$]")
    # axs[1, 0].set_ylim(0, 5)

    plot_velocity_difference(h_opt_velocity, preferential_speed, axs[1, 1])

    fig.subplots_adjust(hspace=0.05)
    fig.savefig("variability_data.png")
    plt.show()


def plot_data(data):

    cw = data["cw"]
    cw_opt_velocity = data["cw_opt_velocity"]
    preferential_speed = data["preferential_speed"]
    lyapunov_exponent = data["lyapunov_exponent"]
    std_angles = data["std_angles"]
    h_5_to_59_percentile = data["h_5_to_59_percentile"]
    mean_com_acceleration = data["mean_com_acceleration"]
    mean_emg = data["mean_emg"]

    plot_energenic_data(cw, cw_opt_velocity, mean_emg, preferential_speed)

    plot_variability(lyapunov_exponent, std_angles, preferential_speed)

    plot_stability(h_5_to_59_percentile, mean_com_acceleration, preferential_speed)
