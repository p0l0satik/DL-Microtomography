import numpy as np
import matplotlib.pyplot as plt
from pyrfc3339 import generate

import pyvista as pv

from perlin_noise import PerlinNoise


# ??
point_reg2 = np.array((-120, 120))[None,:]
point_reg3 = np.array((120, -120))[None,:]
point_reg4 = np.array((120, 120))[None,:]
point_reg5 = np.array((-50, -50))[None,:]
point_reg6 = np.array((-50, 50))[None,:]
point_reg7 = np.array((50, -50))[None,:]
point_reg8 = np.array((50, 50))[None,:]
# ??



class structure_3l:
    def __init__(self, xx, yy, d, rho, Z):
        self.xx = xx    # meshgrid
        self.yy = yy
        self.d = d      #
        self.rho = rho
        self.Z = Z
        self.Z_id = [ int(z+1) for z,_ in enumerate(Z) ]
        self.drho = np.array(d)*np.array(rho)[:, None, None]
        self.Nx = self.xx.shape[0]
        self.Ny = self.xx.shape[1]

    def generate_layer(var, d_max, xx, Nx, Ny, seed):
        #layer2 - Au
        d = np.zeros_like(xx)

        # generate noise
        # noise = PerlinNoise(octaves=2, seed=seed_pair[0]) # provide seed pair for layer2, layer3
        noise = PerlinNoise(octaves=2, seed=seed) # provide seed pair for layer2, layer3

        xpix, ypix = Nx, Ny
        pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
        pic = np.array(pic)

        # quantize noise
        d = pic 
        d[d<0] = 0
        # d2 = 20*d2  #10
        # d2 = d2.astype(np.int)*5 #10 #*5
        d = var[0]*d  #10
        d = d.astype(np.int)*var[1] #10 #*5

        d = d*d_max/(np.max(d)-np.min(d))
        d = d.astype(np.int)
        print("\nd2_thikness: ", (np.min(d), np.max(d)), '\n', np.unique(d) )
        return d

    @classmethod
    def create_test_structure(cls, Nx, Ny, seed_pair):

        x = np.linspace(0, Nx-1, Nx) # coords
        #x = np.linspace(1-Nx//2, Nx//2, Nx)
        
        y = np.linspace(0, Ny-1, Ny)

        xx,yy = np.meshgrid(x,y) # make silicon platform

        #      Si    Au    Al   
        rho = (2.65, 19.3, 2.7) # material density
        Z =   (14,   79,   13)  # atomic number
        Z_id = (1, 2, 3)        # material_id

        # base (layer1)
        d1 = np.ones_like(xx)
        
        d2 = structure_3l.generate_layer((20, 5), 15, xx, Nx, Ny, seed_pair[0])
        # layer3 - Al
        d3 = structure_3l.generate_layer((7, 10), 150, xx, Nx, Ny, seed_pair[1])

        return cls(xx, yy, (d1, d2, d3), rho, Z)


    def plot_3d(self):
        pv.set_plot_theme('document')
        grid1 = pv.StructuredGrid(self.xx, self.yy, self.d[0])
        grid1.point_data['scalars'] = np.average(self.Z_id [0])
        grid2 = pv.StructuredGrid(self.xx, self.yy, self.d[0]+self.d[1])
        grid2.point_data['scalars'] = np.average(self.Z_id [1])
        grid3 = pv.StructuredGrid(self.xx, self.yy, self.d[0]+self.d[1]+self.d[2])
        grid3.point_data['scalars'] = np.average(self.Z_id[2])

        pl = pv.Plotter()
        pl.add_mesh(grid1)
        pl.add_mesh(grid2, opacity=0.85)
        pl.add_mesh(grid3, opacity=0.95)
        pl.show()


    def calc_parameters(self, point):
        ind_x = int(self.xx.shape[0]*(point[:,0] - np.min(self.xx))/(np.max(self.xx) - np.min(self.xx)))
        ind_y = int(self.xx.shape[1]*(point[:,1] - np.min(self.yy))/(np.max(self.yy) - np.min(self.yy)))

        # # #   forms Nx X Ny X 5 array with all parameters at point (x,y)
        d3 = self.d[2][ind_x, ind_y];   # - self.d[1][ind_x, ind_y]
        d2 = self.d[1][ind_x, ind_y]

        rho3 = self.rho[2]
        rho2 = self.rho[1]

        m3 = d3*rho3**(0.91)
        m2 = d2*rho2**(0.91)

        Z3 = self.Z[2]
        Z2 = self.Z[1]

        # base signal for a layer (~scattering coefficient of material which ~Z)
        IB1 = 0.49
        IB2 = 0.9
        IB3 = 0.47

        A3 = self.A(Z3)
        A2 = self.A(Z2)

        layers_num = 3

        if d2 == 0:
            #print('Middle layer absent; renumbering:')
            d2 = d3
            d3 = 0

            m2 = m3
            m3 = 0

            A2 = A3
            A3 = 0

            Z2 = Z3
            Z3 = 0

            rho2 = rho3
            rho3 = 0

            layers_num = 2

        elif d2**2 + d3**2 == 0:
            layers_num = 1

        res = np.array((m3, m2, A3, A2, IB2, IB1))
        return res, layers_num


    def A(self, Z):
        res = 1.0/(0.49 * np.exp(-0.022*(Z + 2)))
        return res

    def psi(self, m, E):
        res = m/(74.0*E**1.55 - m)
        return res

    def E(self, E0, psi3):
        res = E0*np.exp(-(psi3**0.45))
        return res

    def xi(self, A, psi):
        res = 1.0 - np.exp(-A*psi)
        return res

    def phi(self, xi3, xi2):
        res = xi2 - xi2*xi3
        return res


    def calc_signal_point(self, point, E0):
        res = np.zeros((point.shape[0], E0.shape[0]))
        parameters, layers_num = self.calc_parameters(point)


        ind_x = int(self.xx.shape[0]*(point[:,0] - np.min(self.xx))/(np.max(self.xx) - np.min(self.xx)))
        ind_y = int(self.xx.shape[1]*(point[:,1] - np.min(self.yy))/(np.max(self.yy) - np.min(self.yy)))

        m3 = parameters[0]
        m2 = parameters[1]
        A3 = parameters[2]
        A2 = parameters[3]
        IB2 = parameters[4]
        IB1 = parameters[5]

        psi3 = self.psi(m3, E0)
        E2 = self.E(E0, psi3)
        psi2 = self.psi(m2, E2)

        xi3 = self.xi(A3, psi3)
        xi2 = self.xi(A2, psi2)

        phi = xi2 - xi2*xi3

        res = IB1*(1.0-phi) + IB2*phi
        return res, layers_num

    def calc_signal(self, E0, noise_level = 0):
        image = np.zeros((self.Nx, self.Ny, E0.shape[0]))
        print("img shape", image.shape)
        coords = np.array([self.xx, self.yy])
        layers_num = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx-1):
            for j in range(self.Ny-1):
                point = coords[:,i,j][None,:]
                image[i, j, :], layers_num[i,j] = self.calc_signal_point(point, E0)

        image = image + noise_level*image*np.random.randn(image.shape[0], image.shape[1], image.shape[2])


        return image, layers_num


    def signal_3l_regions(self, E0, nl = 0, filename = None):
        I2_mod, _= self.calc_signal_point(point_reg2, E0)
        I3_mod, _= self.calc_signal_point(point_reg3, E0)
        I4_mod, _= self.calc_signal_point(point_reg4, E0)
        I5_mod, _= self.calc_signal_point(point_reg5, E0)
        I6_mod, _= self.calc_signal_point(point_reg6, E0)
        I7_mod, _= self.calc_signal_point(point_reg7, E0)
        I8_mod, _= self.calc_signal_point(point_reg8, E0)

        I2 = I2_mod + I2_mod * nl * np.random.randn(E0.shape[0])
        I3 = I3_mod + I3_mod * nl * np.random.randn(E0.shape[0])
        I4 = I4_mod + I4_mod * nl * np.random.randn(E0.shape[0])
        I5 = I5_mod + I5_mod * nl * np.random.randn(E0.shape[0])
        I6 = I6_mod + I6_mod * nl * np.random.randn(E0.shape[0])
        I7 = I7_mod + I7_mod * nl * np.random.randn(E0.shape[0])
        I8 = I8_mod + I8_mod * nl * np.random.randn(E0.shape[0])

        err2 = np.sqrt(np.sum((I2 - I2_mod)**2)/np.sum(I2_mod**2))*100
        err3 = np.sqrt(np.sum((I3 - I3_mod)**2)/np.sum(I3_mod**2))*100
        err4 = np.sqrt(np.sum((I4 - I4_mod)**2)/np.sum(I4_mod**2))*100
        err5 = np.sqrt(np.sum((I5 - I5_mod)**2)/np.sum(I5_mod**2))*100
        err6 = np.sqrt(np.sum((I6 - I6_mod)**2)/np.sum(I6_mod**2))*100
        err7 = np.sqrt(np.sum((I7 - I7_mod)**2)/np.sum(I7_mod**2))*100
        err8 = np.sqrt(np.sum((I8 - I8_mod)**2)/np.sum(I8_mod**2))*100

        print('Error (region 2) = '+str(err2))
        print('Error (region 3) = '+str(err3))
        print('Error (region 4) = '+str(err4))
        print('Error (region 5) = '+str(err5))
        print('Error (region 6) = '+str(err6))
        print('Error (region 7) = '+str(err7))
        print('Error (region 8) = '+str(err8))

        fig = plt.figure()
        plt.plot(E0, I2, 'r*-', label='I2')
        plt.plot(E0, I3, 'b^--', label='I3')
        plt.plot(E0, I4, 'g<-.', label='I4')
        plt.plot(E0, I5, 'b1:', label='I5')
        plt.plot(E0, I6, 'y8-', label='I6')
        plt.plot(E0, I7, 'mo--', label='I7')
        plt.plot(E0, I8, 'ks-.', label='I8')
        plt.legend()
        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename)
        #plt.show()

        return I2, I3, I4, I5, I6, I7, I8
    
    def parameters_3l_regions(self):
        x2_gt = self.calc_parameters(point_reg2)
        x3_gt = self.calc_parameters(point_reg3)
        x4_gt = self.calc_parameters(point_reg4)
        x5_gt = self.calc_parameters(point_reg5)
        x6_gt = self.calc_parameters(point_reg6)
        x7_gt = self.calc_parameters(point_reg7)
        x8_gt = self.calc_parameters(point_reg8)
        print('x2_gt='+str(x2_gt))
        print('x3_gt='+str(x3_gt))
        print('x4_gt='+str(x4_gt))
        print('x5_gt='+str(x5_gt))
        print('x6_gt='+str(x6_gt))
        print('x7_gt='+str(x7_gt))
        print('x8_gt='+str(x8_gt))

        return x2_gt, x3_gt, x4_gt, x5_gt, x6_gt, x7_gt, x8_gt
