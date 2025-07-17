from dataset import TerrainDataset
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import sys

dataset = sys.argv[-1]
print(dataset)
dataset = TerrainDataset(dataset)

print("Length", len(dataset))

print("Checking NaNs")
for i, el in enumerate(dataset):
    without_nan = dataset[0]['without_nan']
    for y in without_nan:
        for x in y:
            if isnan(x):
                raise Expection("isnan")
print("Done, no NaNs")

fig = plt.figure()

closed = False

def handle_close(evt):
    global closed
    closed = True

def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True

fig.canvas.mpl_connect('close_event', handle_close)
for i, el in enumerate(dataset):
    plt.imshow(el['without_nan'])
    plt.draw()
    waitforbuttonpress()
    plt.imshow(el['mask'])
    plt.draw()
    waitforbuttonpress()
