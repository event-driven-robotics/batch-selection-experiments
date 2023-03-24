import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from model import EV_RotNet

import os
import copy


def load_data(dataset_path, dataset_name, batch_size, shuffle):
    if type(dataset_name) is str:
        dataset_name = [dataset_name]

    paths = [dataset_path + name for name in dataset_name]

    bins = torch.cat([torch.load(path + "_bins.pt") for path in paths])
    rotations = torch.cat([torch.load(path + "_rotations.pt") for path in paths])

    dataset = torch.utils.data.TensorDataset(bins, rotations)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )
    loader_len = len(loader)

    return loader, loader_len


def event_network(
    train_dataset_path,
    train_dataset_name,
    test_dataset_path,
    test_dataset_name,
    model_path,
    tensorboard_path,
    device_name,
    batch_size,
    n_epochs,
    lr,
):
    device = torch.device(device_name)

    print("\nLoading train data ...\n")

    train_loader, train_loader_len = load_data(
        train_dataset_path,
        train_dataset_name,
        batch_size,
        True,
    )

    do_test = test_dataset_path and test_dataset_name
    if do_test:
        print("Loading test data ...\n")

        test_loader, test_loader_len = load_data(
            test_dataset_path, test_dataset_name, 1, False
        )

    print("Loading model ...\n")

    net = EV_RotNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    mse = MSELoss()
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[600, 1000], gamma=0.5
    # )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)

        init_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_net = checkpoint["best_net"]

        net.load_state_dict(checkpoint["last_net"])
        net.train()
    else:
        init_epoch = 1
        best_loss = 1e10
        best_net = net.state_dict()

    writer = SummaryWriter(tensorboard_path)

    print("Starting training\n")

    for epoch in range(init_epoch, n_epochs + 1):
        print("Epoch {} of {} ...".format(epoch, n_epochs))

        train_loss = 0.0

        for data in train_loader:
            bins, gt = (
                data[0].to(device),
                data[1].to(device),
            )

            optimizer.zero_grad()
            pred = net(bins)

            loss = mse(gt, pred)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= train_loader_len
        writer.add_scalar("Loss/train", train_loss, epoch)

        if not (epoch % 4):
            if do_test:
                test_loss = 0.0

                with torch.no_grad():
                    for data in test_loader:
                        bins, gt, mask = (
                            data[0].to(device),
                            data[1].to(device),
                        )

                        pred = net(bins)

                        test_loss = mse(gt, pred)

                test_loss /= test_loader_len

                writer.add_scalar("Loss/test", test_loss, epoch)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_net = copy.deepcopy(net.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "last_net": net.state_dict(),
                    "best_loss": best_loss,
                    "best_net": best_net,
                },
                model_path,
            )

        # scheduler.step()

    writer.flush()
    print("Training finished")

    writer.close()


def event_network(
    train_dataset_path,
    train_dataset_name,
    test_dataset_path,
    test_dataset_name,
    model_path,
    tensorboard_path,
    device_name,
    batch_size,
    n_epochs,
    lr,
):
    device = torch.device(device_name)

    print("\nLoading train data ...\n")

    train_loader, train_loader_len = load_data(
        train_dataset_path,
        train_dataset_name,
        batch_size,
        True,
    )

    do_test = test_dataset_path and test_dataset_name
    if do_test:
        print("Loading test data ...\n")

        test_loader, test_loader_len = load_data(
            test_dataset_path, test_dataset_name, 1, False
        )

    print("Loading model ...\n")

    net = EV_RotNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    mse = MSELoss()
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[600, 1000], gamma=0.5
    # )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)

        init_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        best_net = checkpoint["best_net"]

        net.load_state_dict(checkpoint["last_net"])
        net.train()
    else:
        init_epoch = 1
        best_loss = 1e10
        best_net = net.state_dict()

    writer = SummaryWriter(tensorboard_path)

    print("Starting training\n")

    for epoch in range(init_epoch, n_epochs + 1):
        print("Epoch {} of {} ...".format(epoch, n_epochs))

        train_loss = 0.0

        for data in train_loader:
            bins, gt = (
                data[0].to(device),
                data[1].to(device),
            )

            optimizer.zero_grad()
            pred = net(bins)

            loss = mse(gt, pred)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= train_loader_len
        writer.add_scalar("Loss/train", train_loss, epoch)

        if not (epoch % 4):
            if do_test:
                test_loss = 0.0

                with torch.no_grad():
                    for data in test_loader:
                        bins, gt, mask = (
                            data[0].to(device),
                            data[1].to(device),
                        )

                        pred = net(bins)

                        test_loss = mse(gt, pred)

                test_loss /= test_loader_len

                writer.add_scalar("Loss/test", test_loss, epoch)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_net = copy.deepcopy(net.state_dict())

            torch.save(
                {
                    "epoch": epoch,
                    "last_net": net.state_dict(),
                    "best_loss": best_loss,
                    "best_net": best_net,
                },
                model_path,
            )

        # scheduler.step()

    writer.flush()
    print("Training finished")

    writer.close()

