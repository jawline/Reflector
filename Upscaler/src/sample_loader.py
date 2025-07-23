from math import nan, isnan, isnan
from array import array
from struct import pack, unpack
from functools import partial
from torch import tensor, FloatTensor


def tell_width_height(file):
    with open(file, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise "Error: File too short, could not extract header"
        width, height, scale_z = unpack("<hhf", header)
        for record in iter(partial(file.read, 4), b""):
            values = unpack("<f", record)
        return width, height


class Sample:
    def __init__(self, width, height, scale_z, data):
        self.width = width
        self.height = height
        self.scale_z = scale_z
        self.data = data

    def print(self):
        print(f"Width {self.width} Height {self.height} Scale Z {self.scale_z}")
        for elt in self.data:
            print(elt)

    def tensor(self, x, y, width, height, replace_nan_with=nan):
        out = []
        for ly in range(y, y + height):
            row = []
            for lx in range(x, x + width):
                elt = nan
                if ly < self.height and lx < self.width:
                    elt = self.data[(ly * self.width) + lx]
                if isnan(elt):
                    elt = replace_nan_with
                row.append(elt)
            out.append(row)

        result = FloatTensor(out).reshape((1, width, height))
        return result


def load_sample(file):
    data = array("d")
    with open(file, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise "Error: File too short, could not extract header"
        width, height, scale_z = unpack("<hhf", header)
        for record in iter(partial(file.read, 4), b""):
            values = unpack("<f", record)
            data.extend(values)

    data_len = len(data)

    if data_len != (width * height):
        raise Exception(
            f"Error: width and height don't match size of file {width} {height} {scale_z} {data_len}"
        )

    for elt in data:
        if elt < 0.0:
            raise Exception("data does not look normalized")
        if elt > 1.0:
            raise Exception("data does not look normalized")

    return Sample(width, height, scale_z, data)
