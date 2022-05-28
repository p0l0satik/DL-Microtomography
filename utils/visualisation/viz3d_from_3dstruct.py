import numpy as np
import matplotlib.pyplot as plt
# setting path
sys.path.append('../../')
from utils.generation.object import *

import pyvista as pv
from pyvista import examples

import os
import time

gold = 1
aluminium = 2
empty = 0


def slice_model(model, layer_height = 1):
    #H, W = model.shape[0:2]
    H, W = model.shape[-2:]

    #max_height = int(np.ceil(np.max(model[1] + model[2]) / layer_height))
    max_height = int(np.ceil( (150+15)/ layer_height))
    #print("max_height", max_height)

    zz = np.arange(1, max_height+1, layer_height)
    #print("zz", len(zz))
    #print(zz)

    sliced = np.zeros((H, W, max_height))
    for row in range(H):
        for pixel in range(W):
            # print(row, pixel)
            gold_thic = int(np.ceil(int(model[-2][row][pixel] / layer_height)))
            al_thic = int(np.ceil(int(model[-1][row][pixel] / layer_height)))
            zero_thic = int(max_height) - gold_thic - al_thic
            sliced[row][pixel] = np.array([gold] * gold_thic + [aluminium] * al_thic + [empty] * zero_thic)
    sliced = np.transpose(sliced, (2, 0, 1))
    return sliced, zz


def viz_volume(volume, zz, add_base=True, viz_type="volume"):
    vol = volume.copy()
    vol = np.swapaxes(vol, 0, 1)
    vol = np.swapaxes(vol, 1, 2)
    #print(vol.shape)

    vol[vol==1.]=128
    vol[vol==2.]=255

    if add_base:
        si_base = np.zeros((vol.shape[0], vol.shape[1], 1))
        si_base[:,:,:] = 90
        #print(si_base.shape)

        vol = np.concatenate((si_base, vol), axis =-1)

    pv.set_plot_theme('document') #white background
    pl = pv.Plotter()

    if viz_type == "mesh":
        X, Y, Z = np.meshgrid(sample.xx[0,:], sample.yy[:,0], zz)
            #filename = 'vol_data.npz'
            #np.savez(filename, X = X, Y = Y, Z = Z, rho = vol)
            #print(f'{filename} saved')

        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data['values'] = vol.flatten(order="F")# flatten()

        pl.add_mesh(grid,  opacity="linear")
        #grid.plot(cmap="viridis", opacity=0.4)

    elif viz_type == "volume":
        
        # Uniform Grid
        grid = pv.UniformGrid()                        # Create the spatial reference
        grid.dimensions = np.array(vol.shape) + 1      # Set the grid dimensions
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)

        grid.cell_data["active_scalars"] = vol.flatten(order="F").astype('uint8')  # Flatten the array! # Add the data values to the cell data
        #print(type(grid))

        pl.add_volume(grid,  opacity="linear")

    else:
        print("Specify viz_type : -volume- or -mesh-")

    pl.show()



# example   ################################

filepath = "compare_data/test4k/struct/"
filepath = "compare_data/test4k/rec3d_struct/half_3d_ep20/"
#filepath = "compare_data/train10k/struct/"

file_list = sorted(os.listdir(filepath))

id_list = sorted([file.split("_")[-1] for file in file_list])
print("here", len(id_list))
#[print(i,'-',f) for i,f in zip(id_list, file_list)]






# # # # Creating the model object
sample = structure_3l.create_test_structure(128, 128, seed_pair=(777, 1001))
d = np.array(sample.d)
_, zz = slice_model(d, layer_height = 10)
zz = zz[:-1]
print(len(zz))



for filename in file_list:

    print("\n", filename)

    volume = np.load(filepath + filename)
    print(volume.shape)
    print(np.unique(volume))

    # print viz volume
    viz_volume(volume, zz, add_base=True, viz_type="volume")
#'''