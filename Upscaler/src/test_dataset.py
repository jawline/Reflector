from dataset import TerrainDataset
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import sys
from time import sleep
from plttool import wait_for_key_press

dataset = sys.argv[-1]
print(dataset)
dataset = TerrainDataset(dataset)

print("Length", len(dataset))

print("Checking NaNs")
for i, el in enumerate(dataset):
    print("Next elt")
    without_nan = dataset[0]["without_nan"][0]
    for y in without_nan:
        for x in y:
            if isnan(x):
                raise Expection("isnan")
print("Done, no NaNs")

fig = plt.figure()


fig.canvas.mpl_connect("close_event", handle_close)
for i, el in enumerate(dataset):
    image, mask = el["without_nan"][0], el["mask"][0]
    fig, ax = plt.subplot_mosaic([["image", "mask"]], figsize=(7, 7))
    ax["image"].imshow(image, vmin=0, vmax=1)
    ax["mask"].imshow(mask, vmin=0, vmax=1)
    plt.show()
