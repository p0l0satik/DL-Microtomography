import numpy as np
import matplotlib.pyplot as plt
from object import *

import pyvista as pv
from pyvista import examples

import os
import time

gold = 1
aluminium = 2
empty = 0

# polosatik
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
            gold_thic = int(np.ceil(int(model[1][row][pixel] / layer_height)))
            al_thic = int(np.ceil(int(model[2][row][pixel] / layer_height)))
            zero_thic = int(max_height) - gold_thic - al_thic
            sliced[row][pixel] = np.array([gold] * gold_thic + [aluminium] * al_thic + [empty] * zero_thic)
    sliced = np.transpose(sliced, (2, 0, 1)).astype(np.uint8)
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



dataset_name = "dataset1" #dataset version

os.makedirs(dataset_name, exist_ok=True)

os.makedirs(dataset_name +"/3d_structures", exist_ok=True)
os.makedirs(dataset_name +"/structures", exist_ok=True)
os.makedirs(dataset_name +"/scans", exist_ok=True)


layer2_seeds = np.arange(200,300) #(1,1000) --> produce 100*100 sample structures
#print(layer2_seeds)
layer3_seeds = np.arange(200,300) #(1,1000)
#print(layer2_seeds)

seed_pairs = []
for l2 in layer2_seeds:
    for l3 in layer3_seeds:
        seed_pairs.append( (l2, l3) )

vis = False

for i, seed_pair in enumerate(seed_pairs):
    #print(i)
    t0 = time.time()
    #print(seed_pair)

    # # # # Creating the model object
    
    file = "/struct_{:0>5d}.npy".format(i)
    sample = structure_3l.create_test_structure(128, 128, seed_pair)
    #sample.plot_3d()

    d = np.array(sample.d).astype(np.uint8)

    #polosatik func
    volume, zz = slice_model(d, layer_height = 5)
    volume = volume[:-1, :, :]
    # check volume here
    #print("\n 3d structure")
    #print(volume.shape, volume.dtype, type(volume))
    #u, c = np.unique(volume, return_counts=True)
    #print(u)
    #print(c)
    #viz_volume(volume, zz, add_base=True, viz_type="volume")

    file = "/3d_struct_{:0>5d}.npy".format(i)
    np.save(dataset_name +"/3d_structures" +file, 
            volume, allow_pickle=True)

    
    d = np.array(sample.d)
    #print("\n 2d struct ")
    #print(d.shape, d.dtype, type(d))
    np.save(dataset_name +"/structures" +file, 
            d[1:,:,:], allow_pickle=True)

    # # # # Plotting it
    #sample.plot_3d()

    # # # # Defining the initial electron probe energies (E0)
    E_min = 5.0
    E_max = 35.0
    E_step = 1 #min step = 1 kEv
    N_E = (E_max-E_min)//E_step 

    E0 = np.arange(E_min, E_max+1, E_step) #E0 = np.linspace(E_min, E_max, N_E)

    noise_level =1e-2

    #I2,I3,I4,I4,I6,I7,I8 = sample.signal_3l_regions(E0, noise_level, filename="test_img.png")

    images, layers_num = sample.calc_signal(E0, noise_level)
    images = images.astype(np.float16)
    #print("\ninput")
    #print(images.shape, images.dtype, type(images))

    file = "/scan_{:0>5d}.npy".format(i)
    np.save(dataset_name +"/scans" + file, 
            images, allow_pickle=True)

    if vis:
        for j in range(images.shape[2]):

            img_folder = dataset_name +"/scans" +"/scans_{:0>5d}".format(i)

            os.makedirs(img_folder, exist_ok=True)

            fig = plt.figure()
            plt.imshow(images[:,:,j], vmin=0, vmax =1)
            plt.colorbar()
            plt.title('E_0 = '+str(E0[j]))
            #plt.show()
            fig.savefig( img_folder+"/scan_{}".format(j) )
            plt.close('all')

    t1 = time.time()
    #print("\n gen_time = {:.4f}".format(t1-t0))