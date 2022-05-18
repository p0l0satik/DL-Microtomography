import numpy as np

gold = 1
aluminium = 2
empty = 0

def slice_model(model, layer_height = 1):
    H, W = model.shape[0:2]

    max_height = int(np.ceil(np.max(model[0] + model[1]) / layer_height))

    sliced = np.zeros((H, W, max_height))
    for row in range(H):
        for pixel in range(W):
            # print(row, pixel)
            gold_thic = int(np.ceil(int(model[0][row][pixel] / layer_height)))
            al_thic = int(np.ceil(int(model[1][row][pixel] / layer_height)))
            zero_thic = int(max_height) - gold_thic - al_thic
            sliced[row][pixel] = np.array([gold] * gold_thic + [aluminium] * al_thic + [empty] * zero_thic)
    sliced = np.transpose(sliced, (2, 0, 1))
    return sliced

if __name__ == "__main__":
    model = np.array([[[0,2], [1, 4]],  [[0, 3], [3, 3]]])
    print(slice_model(model))
    model = np.load("dataset0/structures/struct_00002.npy")
    print(slice_model(model).shape)
