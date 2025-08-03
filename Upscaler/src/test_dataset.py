from dataset import TerrainDataset
from math import isnan
import matplotlib.pyplot as plt
import numpy as np
import sys
from time import sleep
from plttool import handle_close, wait_for_key_press
from tqdm import tqdm

dataset = sys.argv[-1]
print(dataset)
dataset = TerrainDataset(dataset)

print("Length", len(dataset))

# print("Checking NaNs")
# for i, el in tqdm(enumerate(dataset)):
#    without_nan = dataset[0]["without_nan"]
#    for y in without_nan:
#        for x in y:
#            if x < 0:
#                print(without_nan)
#                raise Exception("Negative value?", x)
#            if isnan(x):
#                raise Exception("isnan")
# print("Done, no NaNs")

fig = plt.figure()
fig.canvas.mpl_connect("close_event", handle_close)

for i, el in enumerate(dataset):
    image, mask = (
        el["without_nan"],
        el["mask"],
    )
    fig, ax = plt.subplot_mosaic([["image", "mask"]], figsize=(7, 7))
    ax["image"].imshow(image.squeeze(0), vmin=0, vmax=1)
    ax["mask"].imshow(mask.squeeze(0), vmin=0, vmax=1)
    plt.show()
