import numpy as np
import matplotlib.pyplot as plt
from object import *

import os

dataset_name = "dataset0" #dataset version

os.makedirs(dataset_name, exist_ok=True)
os.makedirs(dataset_name +"/structures", exist_ok=True)
os.makedirs(dataset_name +"/scans", exist_ok=True)


layer2_seeds = np.arange(1,30) #(1,1000) --> produce 100*100 sample structures
#print(layer2_seeds)
layer3_seeds = np.arange(1,30) #(1,1000)
#print(layer2_seeds)

seed_pairs = []
for l2 in layer2_seeds:
    for l3 in layer3_seeds:
        seed_pairs.append( (l2, l3) )

vis = False

for i, seed_pair in enumerate(seed_pairs):
    print(i)
    #print(seed_pair)

    # # # # Creating the model object
    
    file = "/struct_{:0>5d}.npy".format(i)
    sample = structure_3l.create_test_structure(128, 128, seed_pair)

    d = np.array(sample.d)
    #print(d.shape)
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
    #print("\ninput test")
    #print(images.shape)

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