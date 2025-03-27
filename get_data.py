import os
import scipy.io as sio
import numpy as np
import pandas as pd
import ezc3d
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from sklearn.metrics import mutual_info_score
from pyomeca import Analogs


def get_energetic_data(data_path):

    filepath = data_path + "/Energetic/K5_data_processed.xlsx"
    data = pd.read_excel(filepath)

    cw = {}
    cw_opt_velocity = {}
    cw_opt = {}
    preferential_speed = {}
    for subject_name in data["Sujet"].unique():
        this_cw = np.array(
            [
                data.loc[data["Sujet"] == subject_name, "Cw_0,5"].values[0],
                data.loc[data["Sujet"] == subject_name, "Cw_0,75"].values[0],
                data.loc[data["Sujet"] == subject_name, "Cw_1"].values[0],
                data.loc[data["Sujet"] == subject_name, "Cw_1,25"].values[0],
                data.loc[data["Sujet"] == subject_name, "Cw_1,5"].values[0],
            ]
        )
        cw[subject_name] = this_cw
        cw_opt_velocity[subject_name] = data.loc[data["Sujet"] == subject_name, "Vit_Cw_bas_P0"].values[0]
        cw_opt[subject_name] = data.loc[data["Sujet"] == subject_name, "Cw_bas_P0"].values[0]
        preferential_speed[subject_name] = data.loc[data["Sujet"] == subject_name, "Vit_pref"].values[0]

    return cw, cw_opt_velocity, cw_opt, preferential_speed


def get_conditions(data_path, preferential_speed):
    path = data_path + "/Kinematic"
    energetic_files = os.listdir(data_path + "/Energetic")
    conditions = {}
    for sub_folder in os.listdir(path):
        try:
            subject_number = f"{int(sub_folder[-2:]):02d}"
        except:
            continue
        condition_filepath = f"{path}/{sub_folder}/{sub_folder}_Cond.mat"
        condition_file = sio.loadmat(condition_filepath)["combinations"]
        conditions[f"Sujet_{subject_number}"] = {
            "0.50": np.nan,
            "0.75": np.nan,
            "1.00": np.nan,
            "1.25": np.nan,
            "1.50": np.nan,
            "preferential_speed": np.nan,
        }
        pref_velocity_idx = np.where(
            np.abs(preferential_speed[f"Sujet_{subject_number}"] - np.array([0.5, 0.75, 1.0, 1.25, 1.5])) < 0.02
        )
        for i in range(condition_file.shape[0]):
            # Keep only the 0 slope conditions
            if condition_file[i, 0] == 0.0:
                velocity = f"{condition_file[i, 1]:.2f}"
                if pref_velocity_idx[0].shape[0] > 0:
                    if np.abs(float(velocity) - preferential_speed[f"Sujet_{subject_number}"] < 0.02):
                        conditions[f"Sujet_{subject_number}"]["preferential_speed"] = f"Cond{i + 1:04d}"
                    elif velocity == "999.00":
                        continue
                elif pref_velocity_idx[0].shape[0] == 0 and velocity == "999.00":
                    conditions[f"Sujet_{subject_number}"]["preferential_speed"] = f"Cond{i + 1:04d}"
                    continue
                conditions[f"Sujet_{subject_number}"][velocity] = f"Cond{i + 1:04d}"


        # See FloEthv for condition order !!!!!!

        # Deduct the 0.5 condition as it is the only one that was not specified
        current_conditions = [conditions[f"Sujet_{subject_number}"][key] for key in conditions[f"Sujet_{subject_number}"].keys()]
        available_files_this_subject = [file for file in energetic_files if file[4:6] == str(subject_number)]
        missing_condition = [file[-12:-4] for file in available_files_this_subject if file[-12:-4] not in current_conditions]
        conditions[f"Sujet_{subject_number}"]["0.50"] = missing_condition[0]

    return conditions


def identify_cycles(entry_names, points, path, sub_folder, subject_number, conditions, nb_frames, velocity):

    force1_index = entry_names.index("force1")
    force2_index = entry_names.index("force2")
    idx1_contact = points[2, force1_index, :] > 10
    idx2_contact = points[2, force2_index, :] > 10
    cycle_start1 = np.where(np.diff(idx1_contact.astype(int)) == 1)
    cycle_start2 = np.where(np.diff(idx2_contact.astype(int)) == 1)

    index_right_heel = entry_names.index("RCAL")
    index_left_heel = entry_names.index("LCAL")
    right_heel_height_when_force1_is_active = np.nanmean(points[2, index_right_heel, idx1_contact])
    left_heel_height_when_force1_is_active = np.nanmean(points[2, index_left_heel, idx1_contact])
    right_heel_height_when_force2_is_active = np.nanmean(points[2, index_right_heel, idx2_contact])
    left_heel_height_when_force2_is_active = np.nanmean(points[2, index_left_heel, idx2_contact])

    # Identify the right leg
    if (
        right_heel_height_when_force1_is_active < left_heel_height_when_force1_is_active
        and right_heel_height_when_force2_is_active > left_heel_height_when_force2_is_active
    ):
        cycle_start = cycle_start1[0]
    elif (
        right_heel_height_when_force1_is_active > left_heel_height_when_force1_is_active
        and right_heel_height_when_force2_is_active < left_heel_height_when_force2_is_active
    ):
        cycle_start = cycle_start2[0]
    else:
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(points[2, force1_index, :], ".r", label="force1")
        axs[0].plot(np.arange(nb_frames)[idx1_contact], points[2, force1_index, idx1_contact], ".k")
        axs[0].plot(
            np.arange(nb_frames)[cycle_start1],
            points[2, force1_index, cycle_start1].reshape(
                -1,
            ),
            "om",
        )
        axs[0].plot(points[2, index_right_heel, :], ".b", label="RCAL")
        axs[0].plot(points[2, index_left_heel, :], ".g", label="LCAL")
        axs[0].set_xlim(0, 1000)

        axs[1].plot(points[2, force2_index, :], ".r", label="force2")
        axs[1].plot(np.arange(nb_frames)[idx2_contact], points[2, force2_index, idx2_contact], ".k")
        axs[1].plot(
            np.arange(nb_frames)[cycle_start2],
            points[2, force2_index, cycle_start2].reshape(
                -1,
            ),
            "om",
        )
        axs[1].plot(points[2, index_right_heel, :], ".b", label="RCAL")
        axs[1].plot(points[2, index_left_heel, :], ".g", label="LCAL")
        axs[1].set_xlim(0, 1000)

        plt.savefig(
            f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6_H_h_5_to_59_percentile.png"
        )
        plt.show()
        raise RuntimeError("I could not identify which force corresponds to which foot. Please see the graph.")

    return cycle_start


def compute_lyapunov(x):
    """
    This implementation of the Lyapunov exponent was generated with ChatGPT.
    It should be verified.
    """

    def time_delay_embedding(x, m, tau):
        """Create time-delay embedding of a 1D signal."""
        N = len(x) - (m - 1) * tau
        return np.array([x[i : i + m * tau : tau] for i in range(N)])

    def average_mutual_information(x, max_lag=100, n_bins=64):
        """
        Compute average mutual information (AMI) for lags up to max_lag.

        Parameters:
        ----------
        x : 1D array
            Time series.
        max_lag : int
            Maximum lag to consider.
        n_bins : int
            Number of bins for histogram-based estimation.

        Returns:
        -------
        ami : array of float
            AMI values for lags from 1 to max_lag.
        """
        ami = []
        x = np.asarray(x)
        x_binned = np.digitize(x, bins=np.histogram_bin_edges(x, bins=n_bins))

        for lag in range(1, max_lag + 1):
            x1 = x_binned[:-lag]
            x2 = x_binned[lag:]
            mi = mutual_info_score(x1, x2)
            ami.append(mi)
        return np.array(ami)

    def false_nearest_neighbors(x, max_dim=10, tau=1, rtol=10, atol=2):
        """
        Estimate the fraction of false nearest neighbors for increasing embedding dimensions.

        Parameters:
        ----------
        x : array-like
            1D time series.
        max_dim : int
            Maximum embedding dimension to try.
        tau : int
            Time delay.
        rtol : float
            Relative distance threshold (usually 10).
        atol : float
            Absolute distance threshold (usually 2).

        Returns:
        -------
        fnn_frac : list of float
            Fraction of false nearest neighbors for each embedding dimension from 1 to max_dim.
        """
        fnn_frac = []
        N = len(x)

        for m in range(1, max_dim + 1):
            X_m = time_delay_embedding(x, m, tau)
            X_m1 = time_delay_embedding(x, m + 1, tau)

            tree = cKDTree(X_m)
            distances, indices = tree.query(X_m, k=2)
            false_neighbors = 0
            total = len(X_m1)

            for i in range(total):
                j = indices[i][1]  # nearest neighbor index
                if i >= len(X_m1) or j >= len(X_m1):
                    continue
                d_m = distances[i][1]
                delta = np.abs(X_m1[i, -1] - X_m1[j, -1])  # extra dimension difference
                ratio = delta / d_m if d_m > 0 else np.inf

                if ratio > rtol or delta > atol:
                    false_neighbors += 1

            fnn_frac.append(false_neighbors / total)

        return fnn_frac

    def rosenstein_lyapunov(x, m=6, tau=1, max_t=50, min_dist=10, fs=1.0, plot=False):
        """
        Estimate the largest Lyapunov exponent using Rosenstein's algorithm.

        Parameters:
        ----------
        x : array-like
            1D time series.
        m : int
            Embedding dimension.
        tau : int
            Time delay.
        max_t : int
            Maximum number of time steps to track divergence.
        min_dist : int
            Minimum time separation between original and neighbor (to avoid autocorrelation).
        fs : float
            Sampling frequency (for exponent in 1/s).
        plot : bool
            Whether to plot the average divergence vs time.

        Returns:
        -------
        lyap_exp : float
            Estimated largest Lyapunov exponent.
        divergence : ndarray
            Mean log divergence over time.
        """
        X = time_delay_embedding(x, m, tau)
        N = len(X)

        tree = cKDTree(X)
        neighbors = np.zeros(N, dtype=int)

        # Find nearest neighbors with temporal constraint
        for i in range(N):
            d, idx = tree.query(X[i], k=2)
            if abs(idx[1] - i) > min_dist:
                neighbors[i] = idx[1]
            else:
                # search manually if closest is too close in time
                distances, indices = tree.query(X[i], k=N)
                for j in indices[1:]:
                    if abs(j - i) > min_dist:
                        neighbors[i] = j
                        break

        divergence = np.zeros(max_t)
        counts = np.zeros(max_t)

        for i in range(N - max_t):
            j = neighbors[i]
            for k in range(max_t):
                if i + k < N and j + k < N:
                    d = np.linalg.norm(X[i + k] - X[j + k])
                    if d > 0:
                        divergence[k] += np.log(d)
                        counts[k] += 1

        valid = counts > 0
        divergence[valid] /= counts[valid]

        # Fit line to linear region (early part)
        t = np.arange(max_t)[valid] / fs
        y = divergence[valid]

        # Use a simple linear fit over a heuristic range
        fit_range = slice(1, min(20, len(t)))
        coef = np.polyfit(t[fit_range], y[fit_range], 1)
        lyap_exp = coef[0]

        if plot:
            plt.plot(t, y, label="Mean log divergence")
            plt.plot(t[fit_range], np.polyval(coef, t[fit_range]), "r--", label=f"Fit: {lyap_exp:.3f} 1/s")
            plt.xlabel("Time [s]")
            plt.ylabel("log divergence")
            plt.title("Rosenstein Lyapunov Exponent")
            plt.legend()
            plt.grid(True)
            plt.show()

        return lyap_exp, divergence

    # Values taken from https://www.sciencedirect.com/science/article/pii/S0966636217310457?casa_token=js00fb9gYPIAAAAA:3MPA7XkQXBc4oLo3gNFb8ep_295GY8wd77fUJpkCCETm96ChokowA3BlBmYWWBd-MQD03PcSqA
    # Find the time delay
    # tau = average_mutual_information(x, max_lag=100, n_bins=64).argmax() + 1
    tau = 10

    # Perform the FNN analysis to find the optimal embedding dimension
    # fnn = false_nearest_neighbors(x, max_dim=18, tau=tau)
    # m = np.argmin(fnn)+1
    m = 5

    lyap, curve = rosenstein_lyapunov(x, m=m, tau=tau, max_t=50, min_dist=50, fs=100, plot=False)
    return lyap

def get_interpolated_cop(cycle_start, platform1_cop, platform2_cop):
    def interpolate(this_cycle):
        y_data = this_cycle[~np.isnan(this_cycle)]
        x_data = np.linspace(0, 1, num=y_data.shape[0])
        interpolation_object = CubicSpline(x_data, y_data)
        return interpolation_object(x_to_interpolate_on)

    nb_cycles = cycle_start.shape[0]
    nb_frames_interp = 100
    cop = np.zeros((4, nb_cycles, nb_frames_interp))
    x_to_interpolate_on = np.linspace(0, 1, num=nb_frames_interp)
    for i_cycle in range(nb_cycles):
        nb_frames = platform2_cop[:2, cycle_start[i_cycle]: cycle_start[i_cycle + 1]].shape[1]
        for i_dim in range(2):
            # left
            this_cycle = platform1_cop[i_dim, cycle_start[i_cycle]: cycle_start[i_cycle + 1]]
            cop[i_dim, i_cycle, :] = interpolate(this_cycle)

            # right
            this_cycle = platform2_cop[i_dim, cycle_start[i_cycle]: cycle_start[i_cycle + 1]]
            cop[i_dim + 2, i_cycle, :] = interpolate(this_cycle)
    return cop


def get_c3d_data(data_path, conditions):
    lyapunov_exponent = {}
    std_angles = {}
    h_5_to_59_percentile = {}
    mean_com_acceleration = {}
    mean_emg = {}
    path = data_path + "/Kinematic"
    for sub_folder in os.listdir(path):
        try:
            subject_number = f"{int(sub_folder[-2:]):02d}"
            lyapunov_exponent[f"Sujet_{subject_number}"] = {}
            std_angles[f"Sujet_{subject_number}"] = {}
            h_5_to_59_percentile[f"Sujet_{subject_number}"] = {}
            mean_com_acceleration[f"Sujet_{subject_number}"] = {}
            mean_emg[f"Sujet_{subject_number}"] = {}
        except:
            continue
        for velocity in ["0.75", "1.00", "1.25", "preferential_speed"]:
            try:
                filepath = f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6.c3d"
            except:
                continue
            c3d = ezc3d.c3d(filepath)
            frame_rate = c3d["header"]["points"]["frame_rate"]
            dt = 1 / frame_rate
            entry_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
            analog_names = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
            points = c3d["data"]["points"]
            nb_frames = points.shape[2]
            cycle_start = identify_cycles(
                entry_names, points, path, sub_folder, subject_number, conditions, nb_frames, velocity
            )
            nb_cycles = cycle_start.shape[0] - 1

            # --- Angular momentum --- #
            print(f"Computing Angular momentum of Sujet_{subject_number}")
            H_index = entry_names.index("H")
            H_norm = np.linalg.norm(points[:3, H_index, :], axis=0)
            percentile_5 = np.percentile(H_norm, 5)
            percentile_95 = np.percentile(H_norm, 95)
            h_5_to_59_percentile[f"Sujet_{subject_number}"][velocity] = percentile_95 - percentile_5

            # plt.figure()
            # plt.plot(H_norm)
            # plt.plot(np.arange(nb_frames),
            #          np.ones((nb_frames, )) * percentile_5)
            # plt.plot(np.arange(nb_frames),
            #          np.ones((nb_frames, )) * percentile_95)
            # plt.savefig(f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6_H_h_5_to_59_percentile.png")
            # plt.close()

            # --- Mean CoM acceleration --- #
            print(f"Computing CoMddot of Sujet_{subject_number}")
            com_index = entry_names.index("CentreOfMass")
            com = points[:3, com_index, :]
            com_acceleration = np.diff(np.diff(com, axis=1), axis=1) / dt**2
            norm_acceleration = np.linalg.norm(com_acceleration, axis=0)
            mean_com_acceleration[f"Sujet_{subject_number}"][velocity] = np.mean(norm_acceleration)

            # --- Angles RMSD --- #
            print(f"Computing STD of Sujet_{subject_number}")
            angles_index = [
                entry_names.index(name)
                for name in ["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"]
            ]
            nb_frames_interp = 100
            angles = np.zeros((18, nb_cycles, nb_frames_interp))
            x_to_interpolate_on = np.linspace(0, 1, num=nb_frames_interp)
            for i_cycle in range(nb_cycles):
                nb_frames = points[:3, 0, cycle_start[i_cycle] : cycle_start[i_cycle + 1]].shape[1]
                for i_angle, angle in enumerate(angles_index):
                    for i_dim in range(3):
                        this_cycle = points[i_dim, angle, cycle_start[i_cycle] : cycle_start[i_cycle + 1]]
                        x_data = np.linspace(0, 1, num=nb_frames)
                        y_data = this_cycle[~np.isnan(this_cycle)]
                        x_data = x_data[~np.isnan(this_cycle)]
                        interpolation_object = CubicSpline(x_data, y_data)
                        angles[i_angle * 3 + i_dim, i_cycle, :] = interpolation_object(x_to_interpolate_on)
            std = np.std(angles, axis=1)
            std_angles[f"Sujet_{subject_number}"][velocity] = np.sum(np.mean(std, axis=1))

            # --- EMG - Energy --- #
            print(f"Computing EMG of Sujet_{subject_number}")
            emg_index = [i_name for i_name, name in enumerate(analog_names) if not name.startswith("Channel")]
            emg_names = [name for name in analog_names if not name.startswith("Channel")]
            if len(emg_index) != 7:
                raise RuntimeError(f"The trial {filepath} does not contain 7 EMG.")
            # emg = c3d["data"]["analogs"][0, emg_index, :]
            # emg_output = np.sum(np.nanmean(np.abs(emg), axis=1))

            emg = Analogs.from_c3d(filepath, suffix_delimiter=".", usecols=emg_names)
            emg_processed = (emg.meca.interpolate_missing_data()
                .meca.band_pass(order=2, cutoff=[10, 425])
                .meca.center()
                .meca.abs()
                .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
                .meca.normalize())
            emg_output = np.sum(np.nanmean(np.array(emg_processed), axis=1))

            mean_emg[f"Sujet_{subject_number}"][velocity] = emg_output

            # # --- Angles Lyapunov --- #
            # print(f"Computing Lyapunov of Sujet_{subject_number}")
            # lyap = []
            # for i_angle, angle in enumerate(angles_index):
            #     for i_dim in range(3):
            #         lyap += [compute_lyapunov(points[i_dim, angle, :])]
            # lyapunov_exponent[f"Sujet_{subject_number}"][velocity] = np.sum(np.array(lyap))

    cop_variability = {}
    path = data_path + "/Kinematic"
    for sub_folder in os.listdir(path):
        try:
            subject_number = f"{int(sub_folder[-2:]):02d}"
            cop_variability[f"Sujet_{subject_number}"] = {}
        except:
            continue
        for velocity in ["0.50", "0.75", "1.00", "1.25"]:
            try:
                filepath = f"data/Energetic/{sub_folder}_K5_{conditions[f'Sujet_{subject_number}'][velocity]}.c3d"
            except:
                continue
            c3d = ezc3d.c3d(filepath, extract_forceplat_data=True)
            frame_rate = c3d["header"]["points"]["frame_rate"]
            dt = 1 / frame_rate

            platform2_force = c3d["data"]["platform"][1]["force"]
            platform1_cop = c3d["data"]["platform"][0]["center_of_pressure"]
            platform2_cop = c3d["data"]["platform"][1]["center_of_pressure"]

            # Identify cycles from platform data (PF2 = right leg for 0° condition)
            idx2_contact = platform2_force[2, :] > 10
            cycle_start = np.where(np.diff(idx2_contact.astype(int)) == 1)[0]

            cop_interpolated = get_interpolated_cop(cycle_start, platform1_cop, platform2_cop)
            std = np.std(cop_interpolated, axis=1)
            cop_variability[f"Sujet_{subject_number}"][velocity] = np.sum(np.mean(std, axis=1))



    return lyapunov_exponent, std_angles, h_5_to_59_percentile, mean_com_acceleration, mean_emg


def get_data(data_path):

    cw, cw_opt_velocity, cw_opt, preferential_speed = get_energetic_data(data_path)

    conditions = get_conditions(data_path, preferential_speed)
    lyapunov_exponent, std_angles, h_5_to_59_percentile, mean_com_acceleration, mean_emg = get_c3d_data(
        data_path, conditions
    )

    data = {
        "cw": cw,
        "cw_opt_velocity": cw_opt_velocity,
        "cw_opt": cw_opt,
        "preferential_speed": preferential_speed,
        "lyapunov_exponent": lyapunov_exponent,
        "std_angles": std_angles,
        "h_5_to_59_percentile": h_5_to_59_percentile,
        "mean_com_acceleration": mean_com_acceleration,
        "mean_emg": mean_emg,
    }

    return data
