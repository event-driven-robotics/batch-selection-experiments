import torch
import numpy as np
from model import EV_RotNet, EV_Imgs
from train import load_data
from torch.nn import MSELoss


def generate_output(
    dataset_path,
    dataset_name,
    model_path,
    last_or_best,
    device_name,
):
    path = dataset_path + dataset_name

    bins = torch.load(path + "_bins.pt")
    gt = torch.load(path + "_rot_diff.pt")

    dataset = torch.utils.data.TensorDataset(bins)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    loader_len = len(loader)

    device = torch.device(device_name)
    net = EV_RotNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint[last_or_best])

    output = np.zeros_like(gt)
    index = 0

    with torch.no_grad():
        for data in loader:
            bins = data[0].to(device)
            output[index] = net(bins).cpu().numpy()
            index += 1

        np.save(path + "_" + model_path[9:-4] + "_r", output)

        print((gt - output).abs().mean())


def test(
    dataset_path,
    dataset_name,
    model_path,
    device_name,
):
    test_loader, test_loader_len = load_data(dataset_path, dataset_name, 1, False, 2)

    device = torch.device(device_name)
    net = EV_RotNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["last_net"])

    mse = MSELoss()
    test_loss = 0.0
    output = torch.zeros((test_loader_len, 3))
    gt1 = torch.zeros((test_loader_len, 3))
    index = 0

    with torch.no_grad():
        for data in test_loader:
            bins, gt = (
                data[0].to(device),
                data[1].to(device),
            )

            pred = net(bins)

            output[index] = pred.cpu()
            gt1[index] = gt.cpu()
            index += 1
            test_loss += mse(gt, pred)

    test_loss /= test_loader_len
    print(test_loss)
    print(((gt1 - output) ** 2).sum().sqrt().mean())
    print((gt1 - output).abs().mean())


def generate_nn_events_output(
    dataset_path,
    dataset_name,
    model_path,
    device_name,
):
    path = dataset_path + dataset_name

    bins = torch.load(path + "_event_imgs.pt")
    gt = torch.tensor(np.load(path + "_n_events.npy"))

    dataset = torch.utils.data.TensorDataset(bins)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    loader_len = len(loader)

    device = torch.device(device_name)
    net = EV_Imgs().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint["best_net"])

    output = np.zeros_like(gt)
    index = 0

    with torch.no_grad():
        for data in loader:
            bins = data[0].to(device)
            output[index] = net(bins).cpu().numpy()
            index += 1

        np.save(path + "_nn_events", output)

        print(((gt - output) / gt).abs().mean())
