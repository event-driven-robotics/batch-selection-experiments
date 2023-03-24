import numpy as np
from bimvee.info import info
from bimvee.importRpgDvsRos import importRpgDvsRos
from bimvee.importIitYarp import importIitYarp


class EventBag:
    def __init__(self, file_path, template, channel="ch0"):
        self.bag = importRpgDvsRos(filePathOrName=file_path, template=template)
        self.channel = channel

        if "dvs" in self.bag["data"][self.channel].keys():
            self.length = len(self.bag["data"][self.channel]["dvs"]["x"])
            self.ts_offset = self.bag["info"]["tsOffsetFromData"]

        if "flowMap" in template[channel]:
            self.flow = self.bag["data"]["ch0"]["flowMap"]

        if "frame" in template[channel]:
            self.depth = self.bag["data"]["ch0"]["frame"]

        if "pose6q" in template[channel]:
            self.pose = self.bag["data"]["ch0"]["pose6q"]

    def __getitem__(self, key):
        return (
            self.bag["data"][self.channel]["dvs"]["y"][key],
            self.bag["data"][self.channel]["dvs"]["x"][key],
            self.bag["data"][self.channel]["dvs"]["ts"][key],
            self.bag["data"][self.channel]["dvs"]["pol"][key],
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        info(self.bag)
        return ""


class EventLog:
    def __init__(self, path, channel="right"):
        self.log = importIitYarp(filePathOrName=path, tsBits=30)
        self.channel = channel
        self.length = len(self.log["data"][self.channel]["dvs"]["x"])

    def __getitem__(self, key):
        return (
            self.log["data"][self.channel]["dvs"]["y"][key],
            self.log["data"][self.channel]["dvs"]["x"][key],
            self.log["data"][self.channel]["dvs"]["ts"][key],
            self.log["data"][self.channel]["dvs"]["pol"][key],
        )

    def __len__(self):
        return self.length

    def __repr__(self):
        info(self.log)
        return ""


class EventData:
    def __init__(self, events, img_dim, ts_offset=0):
        self.x = events[0]
        self.y = events[1]
        self.ts = events[2]
        self.pol = events[3]
        self.length = len(self.x)
        self.dim_x, self.dim_y = (img_dim, img_dim) if type(img_dim) is int else img_dim
        self.ts_offset = ts_offset

    def copy_with(self, new_values):
        copy = self[:]

        for variable in new_values:
            setattr(copy, variable, new_values[variable])

        return copy

    def __iter__(self):
        self.iter_pos = 0
        return self

    def __next__(self):
        if self.iter_pos < self.length:
            item = self.__getitem__(self.iter_pos)
            self.iter_pos += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, key):
        if type(key) is int:
            return (
                self.x[key],
                self.y[key],
                self.ts[key],
                self.pol[key],
            )
        else:
            return EventData(
                (self.x[key], self.y[key], self.ts[key], self.pol[key]),
                (self.dim_x, self.dim_y),
                self.ts_offset,
            )

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            f"Number of events: {self.length}\n"
            + f"Image dimension: ({self.dim_x}, {self.dim_y})"
        )

    def get_coord(self):
        return np.stack((self.x, self.y), axis=1)

    def to_numpy(self):
        return np.stack((self.x, self.y, self.ts, self.pol), axis=1)


class FlowData:
    def __init__(self, x, y, ts):
        self.x = x
        self.y = y
        self.ts = ts
        self.length = len(x)

    def __getitem__(self, key):
        axis = 0 if type(key) is int else 1

        return np.stack([self.x[key], self.y[key]], axis=axis)

    def __len__(self):
        return self.length
