from array import array
from struct import pack, unpack
from functools import partial

class Sample:
    def __init__(self, width, height, scale_z, data):
        self.width = width
        self.height = height
        self.scale_z = scale_z
        self.data = data

def load_sample(file):
    data = array('d')
    with open(file, 'rb') as file:
        header = file.read(8)
        if len(header) != 8:
            raise "Error: File too short, could not extract header"
        width, height, scale_z = unpack("<hhf", header)	
        for record in iter(partial(file.read, 4), b''):
            values = unpack("<f", record)
            data.extend(values)

    data_len = len(data)

    if data_len != (width * height):
        raise Exception(f"Error: width and height don't match size of file {width} {height} {scale_z} {data_len}")

    return Sample(width, height, scale_z, data)
