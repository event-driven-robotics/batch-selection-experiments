from event import EventBag, EventData
import numpy as np
import math as m
import torch
from scipy.spatial.transform import Rotation as Rot
from bimvee.pose import interpolatePoses

path = "/data/Rot1.0/"
name = "arch1_2r"
events_per_input = 400000
img_dim = (256, 256)
bins = 9


def find_first_gt_index(events, gt_events_idx):
    first_index = 0

    for idx in range(len(gt_events_idx)):
        gt_event_idx = gt_events_idx[idx]
        e = events[gt_event_idx - events_per_input : gt_event_idx]
        if len(e) == events_per_input:
            first_index = idx
            break

    return first_index


def find_gt_indexes(events, ts):
    indexes = np.zeros_like(ts, dtype=int)

    for idx in range(len(ts)):
        events_le_ts = events.ts <= ts[idx]
        indexes[idx] = np.where(events_le_ts)[0][-1] + 1

    return indexes


def load_data(path, name):
    template = {
        "ch0": {
            "dvs": "/cam0/events",
            "flowMap": "/cam0/optic_flow",
            "pose6q": "/cam0/pose",
        }
    }

    partial_path = path + name

    bag = EventBag(partial_path + ".bag", template)

    events = EventData(bag[:], img_dim, bag.ts_offset)
    events = events.copy_with({"pol": events.pol * 2 - 1})

    pose = bag.pose

    ts = bag.flow["ts"][1:]
    gt_events_idx = np.load(partial_path + "_gt_events_idx.npy")
    idx1 = find_first_gt_index(events, gt_events_idx)
    ts = ts[idx1:]

    return events, pose, ts, partial_path


def prepare_bins(path, name, adaptive):
    partial_path = path + name

    events = torch.tensor(np.load(partial_path + "_events.npy"))

    gt_ts = np.load(partial_path + "_gt_ts.npy")
    gt_events_idx = np.load(partial_path + "_gt_events_idx.npy")

    idx1 = find_first_gt_index(events, gt_events_idx)

    ts = gt_ts[idx1:]
    events_idx = gt_events_idx[idx1:]
    num_inputs = len(ts)

    n_events = (
        np.load(partial_path + "_n_events.npy")
        if adaptive
        else np.ones(num_inputs) * events_per_input
    )

    in_bins = torch.zeros(num_inputs, bins, img_dim[0], img_dim[1])

    for idx in range(num_inputs):
        print(f"Iter {idx} of {num_inputs - 1}")

        event_idx = events_idx[idx]

        e = events[event_idx - n_events[idx] : event_idx]

        xy = e[:, [0, 1]].long()
        t = e[:, 2]
        p = e[:, 3]
        t_ = (bins - 1) * (t - t[0]) / (t[-1] - t[0])

        for b in range(bins):
            b_idx = (b - 1 < t_) & (t_ <= b + 1)

            b_xy = xy[b_idx]
            b_p = p[b_idx]
            b_t_ = t_[b_idx]

            pos = b_xy[:, 0] * img_dim[0] + b_xy[:, 1]
            b_values = b_p * (1 - abs(b - b_t_))

            in_bins[idx][b].put_(pos, b_values.float(), accumulate=True)

    ### empty space
    events = 0
    gt_ts = 0
    gt_events_idx = 0
    ###

    torch.save(in_bins, partial_path + "_bins.pt")
    torch.save(torch.tensor(ts), partial_path + "_gt_ts.pt")


def prepare_rotations(path, name):
    events, pose, ts, partial_path = load_data(path, name)

    num_inputs = len(ts)
    rot_diff = torch.zeros(num_inputs, 3)
    rot_vel = torch.zeros(num_inputs, 3)

    for idx in range(num_inputs):
        print(f"Iter {idx} of {num_inputs - 1}")

        e = events[events.ts <= ts[idx]][-events_per_input:]

        t1 = e.ts[0]
        t2 = ts[idx]

        interpolated_pose = interpolatePoses(pose, np.array([t1, t2]))

        q1 = Rot.from_quat(interpolated_pose["rotation"][0])
        q2 = Rot.from_quat(interpolated_pose["rotation"][1])

        rotation_diff = (q2 * q1.inv()).as_rotvec()
        ang_vel = rotation_diff / (t2 - t1)

        rot_diff[idx] = torch.Tensor(rotation_diff)
        rot_vel[idx] = torch.Tensor(ang_vel)

    torch.save(rot_diff, partial_path + "_rot_diff.pt")
    torch.save(rot_vel, partial_path + "_rot_vel.pt")


def prepare_adaptive_tensors(path, name):
    events, pose, ts, partial_path = load_data(path, name)

    max_events_per_input = events_per_input + 1
    step = max_events_per_input // 100 + 1
    num_inputs = len(ts)
    n_events = np.zeros(num_inputs, dtype=int)
    rot_diff = torch.zeros(num_inputs, 3)
    rot_vel = torch.zeros(num_inputs, 3)

    for idx in range(num_inputs):
        print(f"Iter {idx} of {num_inputs - 1}")

        e1 = events[events.ts <= ts[idx]][-events_per_input:]

        for ii in range(step, max_events_per_input, step):
            e = e1[-ii:]

            t1 = e.ts[0]
            t2 = ts[idx]

            interpolated_pose = interpolatePoses(pose, np.array([t1, t2]))

            q1 = Rot.from_quat(interpolated_pose["rotation"][0])
            q2 = Rot.from_quat(interpolated_pose["rotation"][1])

            rotation_diff = (q2 * q1.inv()).as_rotvec()

            if np.linalg.norm(rotation_diff) > 0.0174533:
                ang_vel = rotation_diff / (t2 - t1)
                n_events[idx] = ii
                rot_diff[idx] = torch.Tensor(rotation_diff)
                rot_vel[idx] = torch.Tensor(ang_vel)
                break

    np.save(partial_path + "_n_events.npy", n_events)
    torch.save(rot_diff, partial_path + "_rot_diff.pt")
    torch.save(rot_vel, partial_path + "_rot_vel.pt")


def save_event_data_in_numpy(path, name):
    template = {
        "ch0": {
            "dvs": "/cam0/events",
            "flowMap": "/cam0/optic_flow",
        }
    }

    partial_path = path + name

    bag = EventBag(partial_path + ".bag", template)

    events = EventData(bag[:], img_dim, bag.ts_offset)
    events = events.copy_with({"pol": events.pol * 2 - 1})

    gt_ts = bag.flow["ts"][1:]
    gt_events_idx = find_gt_indexes(events, gt_ts)

    events = events.to_numpy()

    np.save(partial_path + "_events", events)
    np.save(partial_path + "_gt_ts", gt_ts)
    np.save(partial_path + "_gt_events_idx", gt_events_idx)


save_event_data_in_numpy(path, name)
prepare_adaptive_tensors(path, name)
prepare_bins(path, name, True)



def generate_event_selection_tensors(path, name):
    partial_path = path + name

    events = np.load(partial_path + "_events.npy")[:, :2].astype(int)
    n_events = np.load(partial_path + "_n_events.npy")
    gt_idx = np.load(partial_path + "_gt_events_idx.npy")

    diff_len = len(gt_idx) - len(n_events)
    gt_idx = gt_idx[diff_len:]

    num_imgs = 16
    step = events_per_input // num_imgs

    imgs = torch.zeros(len(n_events), num_imgs, 256, 256)

    for i, idx in enumerate(gt_idx):
        e = events[idx - events_per_input : idx]

        for j in range(num_imgs):
            ee = e[-(j + 1) * step :]

            imgs[i, j, ee[:, 0], ee[:, 1]] = 1

    torch.save(imgs, partial_path + '_event_imgs.pt')