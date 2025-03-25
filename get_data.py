import os
import scipy.io as sio
import numpy as np
import pandas as pd
import ezc3d
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def get_energetic_data(data_path):

    filepath = data_path + "/Energetic/K5_data_processed.xlsx"
    data = pd.read_excel(filepath)

    cw = {}
    cw_opt = {}
    preferential_speed = {}
    for subject_name in data["Sujet"].unique():
        this_cw = np.array([
            data.loc[data["Sujet"] == subject_name, "Cw_0,5"].values[0],
            data.loc[data["Sujet"] == subject_name, "Cw_0,75"].values[0],
            data.loc[data["Sujet"] == subject_name, "Cw_1"].values[0],
            data.loc[data["Sujet"] == subject_name, "Cw_1,25"].values[0],
            data.loc[data["Sujet"] == subject_name, "Cw_1,5"].values[0],
                  ])
        cw[subject_name] = this_cw
        cw_opt[subject_name] = np.array([
            data.loc[data["Sujet"] == subject_name, "Vit_Cw_bas_P0"].values[0],
            data.loc[data["Sujet"] == subject_name, "Cw_bas_P0"].values[0]
        ])
        preferential_speed[subject_name] = data.loc[data["Sujet"] == subject_name, "Vit_pref"].values[0]

    return cw, cw_opt, preferential_speed


def get_conditions(data_path, preferential_speed):
    path = data_path + "/Kinematic"
    conditions = {}
    for sub_folder in os.listdir(path):
        try:
            subject_number = f"{int(sub_folder[-2:]):02d}"
        except:
            continue
        condition_filepath = f"{path}/{sub_folder}/{sub_folder}_Cond.mat"
        condition_file = sio.loadmat(condition_filepath)["combinations"]
        conditions[f"Sujet_{subject_number}"] = {"0.50": np.nan,
                                                    "0.75": np.nan,
                                                    "1.00": np.nan,
                                                    "1.25": np.nan,
                                                    "1.50": np.nan,
                                                    "preferential_speed": np.nan}
        pref_velocity_idx = np.where(
            np.abs(preferential_speed[f"Sujet_{subject_number}"] - np.array([0.5, 0.75, 1.0, 1.25, 1.5])) < 0.02)
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
    right_heel_height_when_force1_is_active = np.mean(points[2, index_right_heel, idx1_contact])
    left_heel_height_when_force1_is_active = np.mean(points[2, index_left_heel, idx1_contact])
    right_heel_height_when_force2_is_active = np.mean(points[2, index_right_heel, idx2_contact])
    left_heel_height_when_force2_is_active = np.mean(points[2, index_left_heel, idx2_contact])

    # Identify the right leg
    if (right_heel_height_when_force1_is_active < left_heel_height_when_force1_is_active and
            right_heel_height_when_force2_is_active > left_heel_height_when_force2_is_active):
        cycle_start = cycle_start1[0]
    elif (right_heel_height_when_force1_is_active > left_heel_height_when_force1_is_active and
          right_heel_height_when_force2_is_active < left_heel_height_when_force2_is_active):
        cycle_start = cycle_start2[0]
    else:
        raise RuntimeError("I could not identify which force corresponds to which foot.")

    plot_flag = False
    if plot_flag:
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(points[2, force1_index, :], '.r', label="force1")
        axs[0].plot(np.arange(nb_frames)[idx1_contact], points[2, force1_index, idx1_contact], '.k')
        axs[0].plot(np.arange(nb_frames)[cycle_start1], points[2, force1_index, cycle_start1].reshape(-1, ), 'om')
        axs[0].plot(points[2, index_right_heel, :], '.b', label="RCAL")
        axs[0].plot(points[2, index_left_heel, :], '.g', label="LCAL")

        axs[1].plot(points[2, force2_index, :], '.r', label="force2")
        axs[1].plot(np.arange(nb_frames)[idx2_contact], points[2, force2_index, idx2_contact], '.k')
        axs[1].plot(np.arange(nb_frames)[cycle_start2], points[2, force2_index, cycle_start2].reshape(-1, ), 'om')
        axs[1].plot(points[2, index_right_heel, :], '.b', label="RCAL")
        axs[1].plot(points[2, index_left_heel, :], '.g', label="LCAL")

        plt.savefig(
            f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6_H_h_5_to_59_percentile.png")
        plt.close()

    return cycle_start


def get_variability_data(data_path, conditions):
    lyapunov_exponent = {}
    std_angles = {}
    h_5_to_59_percentile = {}
    mean_com_acceleration = {}
    path = data_path + "/Kinematic"
    for sub_folder in os.listdir(path):
        try:
            subject_number = f"{int(sub_folder[-2:]):02d}"
            lyapunov_exponent[f'Sujet_{subject_number}'] = {}
            std_angles[f'Sujet_{subject_number}'] = {}
            h_5_to_59_percentile[f'Sujet_{subject_number}'] = {}
            mean_com_acceleration[f'Sujet_{subject_number}'] = {}
        except:
            continue
        for velocity in ["0.75", "1.00", "1.25", "preferential_speed"]:
            try:
                filepath = f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6.c3d"
            except:
                continue
            c3d = ezc3d.c3d(filepath)
            frame_rate = c3d["header"]["points"]["frame_rate"]
            dt = 1/frame_rate
            entry_names = c3d["parameters"]["POINT"]["LABELS"]['value']
            points = c3d["data"]["points"]
            nb_frames = points.shape[2]
            cycle_start = identify_cycles(entry_names, points, path, sub_folder, subject_number, conditions, nb_frames, velocity)
            nb_cycles = cycle_start.shape[0]-1

            # --- Angular momentum --- #
            H_index = entry_names.index("H")
            H_norm = np.linalg.norm(points[:3, H_index, :], axis=0)
            percentile_5 = np.percentile(H_norm, 5)
            percentile_95 = np.percentile(H_norm, 95)
            h_5_to_59_percentile[f'Sujet_{subject_number}'][velocity] = percentile_95 - percentile_5

            plt.figure()
            plt.plot(H_norm)
            plt.plot(np.arange(nb_frames),
                     np.ones((nb_frames, )) * percentile_5)
            plt.plot(np.arange(nb_frames),
                     np.ones((nb_frames, )) * percentile_95)
            plt.savefig(f"{path}/{sub_folder}/c3d/{sub_folder}_{conditions[f'Sujet_{subject_number}'][velocity]}_processed6_H_h_5_to_59_percentile.png")
            plt.close()

            # --- Mean CoM acceleration --- #
            com_index = entry_names.index("CentreOfMass")
            com  = points[:3, com_index, :]
            com_acceleration = np.diff(np.diff(com, axis=1), axis=1) / dt**2
            norm_acceleration = np.linalg.norm(com_acceleration, axis=0)
            mean_com_acceleration[f'Sujet_{subject_number}'][velocity] = np.mean(norm_acceleration)

            # --- Angles Lyapunov --- #
            angles_index = [entry_names.index(name) for name in ["LHipAngles", "LKneeAngles", "LAnkleAngles", "RHipAngles", "RKneeAngles", "RAnkleAngles"]]

            # --- Angles RMSD --- #
            nb_frames_interp = 100
            angles = np.zeros((18, nb_cycles, nb_frames_interp))
            x_to_interpolate_on = np.linspace(0, 1, num=nb_frames_interp)
            for i_cycle in range(nb_cycles):
                nb_frames = points[:3, 0, cycle_start[i_cycle]: cycle_start[i_cycle+1]].shape[1]
                for i_angle, angle in enumerate(angles_index):
                    for i_dim in range(3):
                        this_cycle = points[i_dim, angle, cycle_start[i_cycle]: cycle_start[i_cycle+1]]
                        x_data = np.linspace(0, 1, num=nb_frames)
                        y_data = this_cycle[~np.isnan(this_cycle)]
                        x_data = x_data[~np.isnan(this_cycle)]
                        interpolation_object = CubicSpline(x_data, y_data)
                        angles[i_angle*3+i_dim, i_cycle, :] = interpolation_object(x_to_interpolate_on)
            std = np.std(angles, axis=1)
            std_angles[f'Sujet_{subject_number}'][velocity] = np.sum(np.mean(std, axis=1))


    return lyapunov_exponent, std_angles, h_5_to_59_percentile, mean_com_acceleration


def get_data(data_path):

    cw, cw_opt, preferential_speed = get_energetic_data(data_path)

    conditions = get_conditions(data_path, preferential_speed)
    lyapunov_exponent, std_angles, h_5_to_59_percentile, mean_com_acceleration = get_variability_data(data_path, conditions)

    data = {
        "cw": cw,
        "cw_opt": cw_opt,
        "preferential_speed": preferential_speed,
        "lyapunov_exponent": lyapunov_exponent,
        "std_angles": std_angles,
        "h_5_to_59_percentile": h_5_to_59_percentile,
        "mean_com_acceleration": mean_com_acceleration,
    }

    return data




















