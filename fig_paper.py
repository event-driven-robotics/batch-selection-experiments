from turtle import title
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os

from matplotlib.colors import LogNorm

folder = '../data/'

models_name = {
    '30K': "f2_30k",
    '100K': "f2_100k",
    'Local': "f1_local_0.001",
    'Rot0.5': "f1_ref_0.001",
    'Rot1.0': "f2_adaptive",
    'RotR': "f2_rand_0.001_cp",
    'Net': "fn_1",
    'Win20': "f1_w20_0.001",
    'Win50': "f1_w50_0.001",
    'Rot1.0r': 'f2_adaptive',
    'RotRr': 'f2_rand_0.001_cp',
    'Rot0.5r': "f1_ref_0.001"
}


def plt_n_events(dataset, models__, models_, add_ylabel, add_legend, file_name):
    ts = torch.load(folder + 'common/' + dataset + '_gt_ts.pt').numpy()

    def plot(model, line_type):
        if model == '30K':
            plt.plot(ts, 30000 * np.ones_like(ts), line_type, label='30K')
        elif model == '100K':
            plt.plot(ts, 100000 * np.ones_like(ts), line_type, label='100K')
        else:
            n_events = np.load(folder + model + '/' + dataset + '_n_events.npy')
            idx = len(n_events) - len(ts)
            ts_plt = ts[:idx] if idx != 0 else ts
            plt.plot(ts_plt, n_events, line_type, label=model)

    for model in models__:
        plot(model, '--')
    for model in models_:
        plot(model, '-')

    plt.xlabel("Time (seconds)", fontsize=16)
    if add_ylabel:
        plt.ylabel("Batch Size (events)", fontsize=16)
    if add_legend:
        plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    ylabels = [str(int(y)) + 'K' for y in ax.get_yticks() / 1000]
    ax.set_yticklabels(ylabels)
    plt.yticks(fontsize=16)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_3d_cloud_events(dataset, first_event, n_events, file_name):
    events = np.load(folder + 'common/' + dataset + "_events.npy")

    ev = events[first_event:first_event + n_events]

    positive = ev[:, 3] == 1
    negative = ~positive

    #fig = plt.figure(figsize=(12, 12))
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(projection='3d')
    yellow = [99.2/255, 90.6/255, 14.15/255]
    blue = [26.7/255, 0.4/255, 32.9/255]
    ax.scatter(ev[:, 2][negative], ev[:, 1][negative], ev[:, 0][negative], s=0.5)
    ax.scatter(ev[:, 2][positive], ev[:, 1][positive], ev[:, 0][positive], s=0.5)

    ax.set_ylabel('X (pixels)', fontsize=12)
    ax.set_zlabel('Y (pixels)', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    plt.xticks([2.62, 2.63, 2.64], fontsize=10)
    plt.yticks(fontsize=12)
    ax.zaxis.set_tick_params(labelsize=12)
    ax.view_init(azim=12, elev=-162)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_img_events(dataset, first_event, n_events, file_name):
    events = np.load(folder + 'common/' + dataset + "_events.npy")

    img = np.zeros((256, 256))
    ev_to_process = events[first_event:first_event + n_events]

    for event in ev_to_process.astype(int):
        img[event[0], event[1]] = event[3]

    plt.imshow(img)
    plt.xlabel('X (pixels)', fontsize=14)
    plt.ylabel('Y (pixels)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def avg_n_events(dataset, models):
    ts = torch.load(folder + 'common/' + dataset + '_gt_ts.pt').numpy()

    avg_events = []

    for model in models:
        n_events = np.load(folder + model + '/' + dataset + '_n_events.npy')
        avg_events.append(n_events.mean())

    return int(sum(avg_events) / len(avg_events))

def err_vs_rot(dataset, selection):
    rot = np.load(folder + 'Rot1.0' + '/' + dataset + '_n_events.npy')
    net = np.load(folder + selection + '/' + dataset + '_n_events.npy')

    return abs(np.log10(rot) - np.log10(net)).mean()

def get_data(dataset, events, selection, model):
    def align_data(data1):
        return data1[len(data1) - len(ts) :] if len(data1) - len(ts) > 0 else data1

    model = model if model == 'GT' else models_name[model]

    ts = torch.load(folder + 'common/' + dataset + "_gt_ts.pt")
    idx = np.load(folder + 'common/' + dataset + "_gt_events_idx.npy")
    data = torch.load(folder + selection + '/' + dataset + "_rot_diff.pt") if model == 'GT' else np.load(folder + selection + '/' + dataset + "_" + model + "_r.npy")

    idx = align_data(idx)
    data = align_data(data)

    if selection == '30K' or selection == '100K':
        n_events = 30000 if selection == "30K" else 100000
    else:
        n_events = np.load(folder + selection + '/' + dataset + "_n_events.npy")

    ts_diff = events[idx - 1][:, 2] - events[idx - n_events - 1][:, 2]
    data[:, 0] = data[:, 0] / ts_diff
    data[:, 1] = data[:, 1] / ts_diff
    data[:, 2] = data[:, 2] / ts_diff

    return data

def err_prediction(dataset, events, selection, model):
    err = abs(get_data(dataset, events, selection, 'GT') - get_data(dataset, events, selection, model)).mean(dim=1)

    return err.numpy()

def err_prediction_all(events_all, selection, model):
    datasets_all = ['room_1r', 'room_2r', 'room_3r']

    return np.concatenate([err_prediction(dataset, events, selection, model) for (dataset, events) in zip(datasets_all, events_all)])

def avg_err_prediction(dataset, selection, model, events=None):
    if type(events) is not np.ndarray:
        events = np.load(folder + 'common/' + dataset + "_events.npy")

    return err_prediction(dataset, events, selection, model).mean()

def avg_norm_err_prediction(dataset, selection, model, events):
    gt = get_data(dataset, events, selection, 'GT')
    pred = get_data(dataset, events, selection, model)

    return (np.linalg.norm(gt - pred, axis=1) / np.linalg.norm(gt, axis=1)).mean()

def plt_boxplot_best_all(dataset, models, skip_models_all, ylimit, file_name, best_or_all):
    events = np.load(folder + 'common/' + dataset + "_events.npy")

    selections = ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50']

    best_errs = []
    best_selections = []
    best_models = []

    all_errs = []
    all_selections = []
    all_models = []

    for model in models:
        err = err_prediction(dataset, events, model, model)
        best_errs += err.tolist()
        best_selections += ["trained"] * len(err)
        best_models += [model] * len(err)

        if model not in skip_models_all:
            err_all = np.concatenate([err_prediction(dataset, events, selection, model) for selection in selections])
            all_errs += err_all.tolist()
            all_selections += ["all"] * len(err_all)
            all_models += [model] * len(err_all)

    if best_or_all == 'best':
        data = {}
        data["error"] = best_errs + all_errs
        data["selection"] = best_selections + all_selections
        data["model"] = best_models + all_models

    df = pd.DataFrame(data)
    sns.boxplot(x="model", y="error", hue="selection", data=df, showfliers=False)
    if ylimit:
        plt.ylim(ylimit)
    plt.xticks(fontsize=8)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_bars_trained(file_name):
    datasets_all = ['room_1r', 'room_2r', 'room_3r']
    events = [np.load(folder + 'common/' + dataset + "_events.npy") for dataset in datasets_all]

    selections = ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50', 'Rot1.0r', 'RotRr']
    errs = []

    for selection in selections:
        errs.append(err_prediction_all(events, selection, selection).mean())

    index = np.argsort(errs)

    bars = plt.bar(np.array(selections)[index], np.array(errs)[index])
    bars[0].set_color('r')
    bars[3].set_color('r')
    plt.xticks(fontsize=8)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()




def plt_boxplot_all(file_name):
    datasets_all = ['room_1r', 'room_2r', 'room_3r']
    events = [np.load(folder + 'common/' + dataset + "_events.npy") for dataset in datasets_all]

    selections = ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50']
    errs = []

    for model in selections:
        err_model = []

        for selection in selections:
            if selection != 'Local' or (selection == 'Local' and model == 'Local'):
                err_model += err_prediction_all(events, selection, model).tolist()

        errs.append(sum(err_model) / len(err_model))

    index = np.argsort(errs)

    bars = plt.bar(np.array(selections)[index], np.array(errs)[index])
    bars[4].set_color('r')
    bars[5].set_color('r')
    #plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_boxplot_worst(selections, file_name):
    datasets = ['room_1r', 'room_2r', 'room_3r']

    errs = []
    scenes = []
    models = []

    for dataset in datasets:
        events = np.load(folder + 'common/' + dataset + "_events.npy")

        for selection in selections:
            err = err_prediction(dataset, events, selection, selection)

            errs += err.tolist()
            scenes += [dataset] * len(err)
            models += [selection] * len(err)

    data = {}
    data["error"] = errs
    data['dataset'] = scenes
    data["selection"] = models

    df = pd.DataFrame(data)
    sns.boxplot(x="dataset", y="error", hue="selection", data=df, showfliers=False)
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_xyz(dataset, selections, models, ylimit, file_name):
    events = np.load(folder + 'common/' + dataset + "_events.npy")

    ts = torch.load(folder + 'common/' + dataset + "_gt_ts.pt")

    fig, axs = plt.subplots(3)

    for selection, model in zip(selections, models):
        data = get_data(dataset, events, selection, model)

        axs[0].plot(ts, data[:, 1], '--' if model == 'GT' else '-', label=model)
        axs[1].plot(ts, data[:, 2], '--' if model == 'GT' else '-')
        axs[2].plot(ts, data[:, 0], '--' if model == 'GT' else '-')

    axs[0].legend()

    axs[0].set_ylabel('x')
    axs[1].set_ylabel('y')
    axs[2].set_ylabel('z')

    axs[0].set_ylim(ylimit[0], ylimit[1])
    axs[1].set_ylim(ylimit[0], ylimit[1])
    axs[2].set_ylim(ylimit[0], ylimit[1])

    axs[0].set_xticks([])
    axs[1].set_xticks([])

    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_correlation(file_name):
    events_dict = {
        'room_1r': np.load(folder + 'common/' + 'room_1r' + "_events.npy"),
        'room_2r': np.load(folder + 'common/' + 'room_2r' + "_events.npy"),
        'room_3r': np.load(folder + 'common/' + 'room_3r' + "_events.npy"),
    }

    def scatter_plt(model, marker):
        for selection in ['30K', '100K', 'Win20', 'Win50', 'Rot0.5', 'Rot1.0', 'RotR', 'Net']:
            events_err = []
            motion_err = []

            for dataset in ['room_2r', 'room_3r']:
                events = events_dict[dataset]

                rot1_err = avg_norm_err_prediction(dataset, 'Rot1.0', model, events)
                gt_n_events = np.load(folder + 'Rot1.0' + '/' + dataset + '_n_events.npy')

                if "30" in selection:
                    n_events = 30000
                elif "100" in selection:
                    n_events = 100000
                else:
                    n_events = np.load(folder + selection + '/' + dataset + '_n_events.npy')

                events_err.append(abs(np.log(gt_n_events) - np.log(n_events)).mean())
                motion_err.append(avg_norm_err_prediction(dataset, selection, model, events) - rot1_err)

            return plt.scatter(events_err, motion_err, marker=marker, label=selection)

    fig, ax = plt.subplots()

    scatter1 = scatter_plt('Rot1.0', 'o')
    legend1 = plt.legend(title="Model Rot1.0", handles=[scatter1], loc='upper right')
    ax = plt.gca().add_artist(legend1)

    scatter2 = scatter_plt('RotR', 'x')
    plt.legend(title="Model RotR", handles=[scatter2], loc='upper left')

    plt.xlabel("No. of Events Deviation from 1 Degree Rotation")
    plt.ylabel("Normalized Error")
    #plt.legend(title="Selection Method")
    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()

def plt_room_boxplot():
    datasets_all = ['room_1r', 'room_2r', 'room_3r']
    events_all = [np.load(folder + 'common/' + dataset + "_events.npy") for dataset in datasets_all]

    selections = ['Rot1.0', 'RotR', 'Win50', 'Rot0.5', 'Win20', '30K', 'Net', '100K', 'Local']
    errs = []
    models = []

    for selection in selections:
        err = err_prediction_all(events_all, selection, selection)

        errs += err.tolist()
        models += [selection] * len(err)

    data = {}
    data["error"] = errs
    data["model"] = models

    df = pd.DataFrame(data)

    sns.set(style="darkgrid")
    palette = {model: "r" if model == "Rot0.5" or model == "Rot1.0" or model == 'RotR' else 'b' for model in selections}

    fig = plt.figure(figsize=(14, 8))
    sns.boxplot(x="model", y="error", data=df, order=np.array(selections), palette=palette, showfliers=False)

    plt.ylabel('Error', fontsize=20)
    plt.xlabel('Model', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('room_boxplot', bbox_inches="tight", dpi=200)
    plt.show()

def plt_ang_velocity_all(file_name):
    dataset_names = ['arch1_1r', 'arch1_2r', 'arch1_3r', 'arch1_4r', 'arch1_5r', 'arch2_1r', 'arch2_2r', 'arch2_3r',
    'arch2_4r', 'arch3_1r', 'arch3_2r', 'arch3_3r', 'room_1r', 'room_2r', 'room_3r']

    vels = []
    datasets = []

    for dataset in dataset_names:
        vel = torch.load(folder + 'common/' + dataset + "_rot_vel.pt").norm(dim=1)

        vels += vel.tolist()
        dataset_name = 'room' if 'room' in dataset else dataset[:-1]
        datasets += [dataset_name] * len(vel)

    data = {}
    data["Angular Velocity (rad/s)"] = vels
    data['Dataset'] = datasets
    df = pd.DataFrame(data)

    fig = plt.figure(figsize=(22, 8))
    sns.set(font_scale=1.9)
    sns.set(style="darkgrid")
    palette = {'b' for dataset in dataset_names + ['room']}
    sns.boxplot(x="Dataset", y="Angular Velocity (rad/s)", data=df, showfliers=False, palette=palette)
    plt.ylabel("Angular Velocity (rad/s)", fontsize=23)
    plt.xlabel('Dataset', fontsize=23)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()

def plt_heatmap_errs():
    datasets_all = ['room_1r', 'room_2r', 'room_3r']
    events_all = [np.load(folder + 'common/' + dataset + "_events.npy") for dataset in datasets_all]

    selections = ['Rot1.0', 'RotR', 'Win50', 'Rot0.5', 'Win20', '30K', 'Net', '100K', 'Local', 'Recursive']
    models = ['Rot1.0', 'RotR', 'Win50', 'Rot0.5', 'Win20', '30K', 'Net', '100K', 'Local']
    num_selections = len(selections)
    num_models = len(models)
    errs = np.zeros((num_selections, num_models))

    for i, selection in enumerate(selections):
        for j, model in enumerate(models):
            if selection != 'Recursive':
                err = err_prediction_all(events_all, selection, model)
                errs[i, j] = err.mean()
            else:
                if 'Rot' in model:
                    err = err_prediction_all(events_all, model + 'r', model + 'r')
                    errs[i, j] = err.mean()

    errs[errs==0] = 100
    annot = errs.copy()
    annot[annot == 100] = -1

    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
    heatmap = sns.heatmap(errs, annot=annot, vmin=0.1, vmax=0.2, cmap=cmap, xticklabels=models, yticklabels=selections, cbar=False)
    heatmap.set(xlabel='Model', ylabel='Selection')
    for tick_label in heatmap.axes.get_yticklabels():
        if 'Rot' in tick_label.get_text():
            tick_label.set_color("red")
    plt.savefig('room_heatmap', bbox_inches="tight", dpi=300)
    plt.show()

def plt_xyz_all(selections, models, ylimit, file_name, zoom):
    datasets_all = ['room_1r', 'room_2r', 'room_3r']
    events_all = [np.load(folder + 'common/' + dataset + "_events.npy") for dataset in datasets_all]

    ts_add = [0, 14, 14 + 15]
    ts = np.concatenate([torch.load(folder + 'common/' + dataset + "_gt_ts.pt") + add for dataset, add in zip(datasets_all, ts_add)])

    fig, axs = plt.subplots(1 if zoom else 3)

    for selection, model in zip(selections, models):
        data = np.concatenate([get_data(dataset, events, selection, model) for dataset, events in zip(datasets_all, events_all)], axis=0)
        if zoom:
            data = data[:zoom, :]
            ts = ts[:zoom]
            axs.plot(ts, data[:, 1], '--' if model == 'GT' else '-', label=model)
        else:
            axs[0].plot(ts, data[:, 1], '--' if model == 'GT' else '-', label=model)
            axs[1].plot(ts, data[:, 2], '--' if model == 'GT' else '-')
            axs[2].plot(ts, data[:, 0], '--' if model == 'GT' else '-')

    if zoom:
        axs.legend(fontsize=14)
        axs.set_xlabel('Time (seconds)', fontsize=16)
        axs.set_ylabel('x (rad/s)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        axs.set_ylim(ylimit[0], ylimit[1])
    else:
        axs[0].legend(loc='upper left', fontsize=12)
        axs[0].set_ylabel('x', fontsize=16)
        axs[0].set_ylim(ylimit[0], ylimit[1])
        axs[1].set_ylabel('y', fontsize=16)
        axs[2].set_ylabel('z', fontsize=16)
        axs[2].set_xlabel('Time (seconds)', fontsize=16)
        axs[1].set_ylim(ylimit[0], ylimit[1])
        axs[2].set_ylim(ylimit[0], ylimit[1])
        axs[0].set_xticks([])
        axs[0].yaxis.set_tick_params(labelsize=16)
        axs[1].set_xticks([])
        axs[1].yaxis.set_tick_params(labelsize=16)
        plt.xlabel('Time (seconds)', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(file_name, bbox_inches="tight", dpi=200)
    plt.show()


####### Section 4 Experiments and Results

##### Section 4.4 Evaluation

### fig:gt_n_events


### second paragraph: average number of events in datasets of fig:gt_n_events
avg_n_events('arch1_1r', ['Rot0.5', 'Rot1.0', 'Win50'])
avg_n_events('room_3r', ['Rot0.5', 'Rot1.0', 'Win20', 'Win50'])

### error vs  rot1.0 selection
e1 = err_vs_rot('room_1r', 'Net')
e2 = err_vs_rot('room_2r', 'Net')
e3 = err_vs_rot('room_3r', 'Net')
print((e1 + e2 + e3) / 3)
e1 = err_vs_rot('room_1r', 'Rot1.0r')
e2 = err_vs_rot('room_2r', 'Rot1.0r')
e3 = err_vs_rot('room_3r', 'Rot1.0r')
print((e1 + e2 + e3) / 3)
e1 = err_vs_rot('room_1r', 'RotRr')
e2 = err_vs_rot('room_2r', 'RotRr')
e3 = err_vs_rot('room_3r', 'RotRr')
print((e1 + e2 + e3) / 3)

### plot rot1.0 vs net selection
plt_n_events('room_1r', ['Rot1.0'], ['Net', 'Rot1.0r'], True, True, 'room_1r_rot1_approximation')
plt_n_events('room_2r', ['Rot1.0'], ['Net', 'Rot1.0r'], False, False, 'room_2r_rot1_approximation')
plt_n_events('room_3r', ['Rot1.0'], ['Net', 'Rot1.0r'], False, False, 'room_3r_rot1_approximation')

### boxplot results

# plt_boxplot_best_all('room_1r', ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50', 'Rot1.0r', 'RotRr'], ['Rot1.0r', 'RotRr'], [-0.08, 0.8], 'room_1r_boxplot')
# plt_boxplot_best_all('room_2r', ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50', 'Rot1.0r', 'RotRr'], ['Rot1.0r', 'RotRr'], [-0.04, 0.4], 'room_2r_boxplot')
# plt_boxplot_best_all('room_3r', ['30K', '100K', 'Local', 'Rot0.5', 'Rot1.0', 'RotR', 'Net', 'Win20', 'Win50', 'Rot1.0r', 'RotRr'], ['Rot1.0r', 'RotRr'], [-0.08, 0.8], 'room_3r_boxplot')

### xyz plot best

plt_xyz('room_1r', ['Rot1.0', 'Rot1.0', 'Win50', 'Net'], ['GT', 'Rot1.0', 'Win50', 'RotR'], [-1.85, 1.85], 'room_1r_xyz')
plt_xyz('room_2r', ['Rot1.0', 'Rot1.0', 'Win50', 'Net'], ['GT', 'Rot1.0', 'Win50', 'RotR'], [-1.5, 1.5], 'room_2r_xyz')
plt_xyz('room_3r', ['Rot1.0', 'Rot1.0', 'Win50', 'Net'], ['GT', 'Rot1.0', 'Win50', 'RotR'], [-1.5, 1.5], 'room_3r_xyz')
plt_xyz('room_1r', ['Rot1.0', 'Local'], ['GT', 'Local'], [-2, 2], 'room_1r_xyz_local')

### correlation 1.0 rotation plot

plt_correlation('err_correlation')

plt_xyz_all(['Rot1.0', '100K', 'Rot1.0', 'Win50', 'Net'], ['GT', '100K', 'Rot1.0', 'Win50', 'RotR'], [-1.85, 1.85], 'room_xyz', 500)

plt_xyz_all(['100K', 'Rot1.0'], ['100K', 'GT'], [-1.85, 1.85], 'room_xyz_100', None)



plt_n_events('room_3r', ['Rot1.0'], ['Net', 'Rot1.0r', 'Local'], False, True, 'room_3r_rot1_approximation')
plt_n_events('room_1r', ['30K', '100K'], ['Local', 'Rot0.5', 'Rot1.0', 'Win20', 'Win50'], True, True, 'room_3r_gt_events')
plt_n_events('room_3r', ['30K', '100K'], ['Rot0.5', 'Rot1.0', 'Win20', 'Win50'], True, True, 'room_3r_gt_events')
plt_n_events('arch1_1r', ['30K', '100K'], ['Rot0.5', 'Rot1.0', 'Win20', 'Win50'], False, False, 'arch1_1r_gt_events')



### Figure 1(a), (b)
plt_3d_cloud_events('room_3r', 2000000, 1000, 'cloud_events_1k')
plt_3d_cloud_events('room_3r', 2000000, 10000, 'cloud_events_10k')

### Figure 1(c), (d)
plt_img_events('room_3r', 2000000, 1000, 'img_events_1k')
plt_img_events('room_3r', 2000000, 10000, 'img_events_10k')

### Figure 5(e)
plt_ang_velocity_all('meanvel')

### Figure 6(a), (b)
plt_n_events('arch1_1r', ['30K', '100K'], ['Local', 'Rot0.5', 'Rot1.0', 'Win20', 'Win50'], True, True, 'arch1_1r_gt_events')
plt_n_events('room_3r', ['Rot1.0'], ['Net', 'Rot1.0r', 'Local'], False, True, 'room_3r_gt_events')

### Figure 8
plt_room_boxplot() # room_boxplot

### Figure 9(a), (b)
plt_xyz_all(['Rot1.0', 'Rot1.0', 'Win50', 'Net', '100K'], ['GT', 'Rot1.0', 'Win50', 'RotR', '100K'], [-1.85, 1.85], 'room_xyz_zoom', 500)
plt_xyz_all(['Rot1.0', 'Rot1.0', 'Win50', 'Net'],         ['GT', 'Rot1.0', 'Win50', 'RotR'],         [-1.85, 1.85], 'room_xyz',      None)

### Figure 10
plt_heatmap_errs() # room_heatmap