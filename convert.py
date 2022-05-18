import numpy as np
from pandas import array
model = np.load("dataset0/structures/struct_00002.npy")
# model = np.load("dataset0/scans/scan_00000.npy")
layer_height = 1
gold = 1
aluminium = 2
empty = 0
H, W = 128, 128
max_height = np.max(model[0] + model[1])
print(max_height)

sliced = np.zeros((H, W, int(np.ceil(max_height / layer_height))))
for row in range(H):
    for pixel in range(W):
        gold_thic = int(np.ceil(int(model[0][row][pixel] / layer_height)))
        al_thic = int(np.ceil(int(model[1][row][pixel] / layer_height)))
        zero_thic = int(max_height) - gold_thic - al_thic
        sliced[row][pixel] = np.array([gold] * gold_thic + [aluminium] * al_thic + [0] * zero_thic)
sliced = np.transpose(sliced, (2, 0, 1))
print(sliced.shape)
