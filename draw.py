import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import polar_from_xyz

# constants
Ri = 17.71e3 # inner radius / mm
Ro = 19.5e3 # outer radius / mm
N_vertices = 4000 # total num of vertices
Volume_i = 4 / 3 * np.pi * Ri ** 3 # volume of LS
rho0 = N_vertices / Volume_i # average density / mm^-3

# 该类在测试时会用到，请不要私自修改函数签名，后果自负
class Drawer:
    def __init__(self, data, geo):
        self.simtruth = data["ParticleTruth"]
        self.petruth = data["PETruth"]
        self.geo = geo["Geometry"]

    def draw_vertices_density(self, fig, ax):
        '''
        draw density of vertices as a function of radius:
        density = density(radius)
        '''
        x = np.array(self.simtruth['x'])
        y = np.array(self.simtruth['y'])
        z = np.array(self.simtruth['z'])

        # radius
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # total number
        N = len(x)

        # extract hist statistics
        # density=True: n = dN/(N * dr)
        n, bins, patches = ax.hist(r, bins=1000, density=True)
        ax.cla()

        # plot
        ax.set_title(r'Volume Density of Vertices $\rho(r)$')
        ax.set_xlabel(r'Radius from Origin $r$ / $R_{LS}$')
        ax.set_xlim(0, Ri)
        ax.set_xticks(np.linspace(0, Ri, 11))
        ax.set_xticklabels(['%.1f' % (i / 10) for i in range(11)])

        ax.set_ylabel(
            r'Volume Density of Vertices $\rho(r)$ / $\rho_0 = {:.2f} \times 10^{{{:.0f}}} mm^{{-3}}$'
            .format(
                *map(float, 
                ('%.2e'%rho0).split('e'))
                ))
        # ax.set_ylim(0, 2 * rho0)
        # ax.set_yticks(np.linspace(0, 2 * rho0, 6))
        # ax.set_yticklabels(['%.1f' % (2 * i / 5) for i in range(6)])
        
        # density = dN / (4 pi r^2 dr) = n / (4 pi r^2) * N
        ax.plot(bins[1:], n / (4 * np.pi * bins[1:] ** 2) * N / rho0)

        

    def draw_pe_hit_time(self, fig, ax):
        '''
        draw histogram of PE hit time:
        #PE = #PE(time)
        '''
        time = np.array(self.petruth['PETime'])
        maxtime = np.max(time)

        # plot
        ax.set_title(r'Histogram of PE Hit Time')
        ax.set_xlabel(r'Hit Time $t$ / $ns$')
        ax.set_xlim(0, maxtime)
        ax.set_xticks(np.linspace(0, maxtime, 11))
        ax.set_xticklabels(['%.0f' % i for i in np.linspace(0, maxtime, 11)])

        ax.set_ylabel(r'Number of PE Hit')

        ax.hist(time, bins=1000, density=False)
        

    def draw_probe(self, fig, ax):
        '''
        draw probe function
        average over all PMTs (Channels)
        probe = probe(theta, r)
        '''
        pt = self.simtruth
        pet = self.petruth
        geo = self.geo

        # Events, Events_i = np.unique(pet['EventID'], return_inverse=True)
        Channels, Channels_i = np.unique(pet['ChannelID'], return_inverse=True)

        pet_geo = np.array(geo[Channels][Channels_i][1:])


        breakpoint()
        
        # ax.hist2d()


if __name__ == "__main__":
    import argparse

    # 处理命令行
    parser = argparse.ArgumentParser()
    parser.add_argument("ipt", type=str, help="Input simulation data")
    parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
    parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
    args = parser.parse_args()

    # 读入模拟数据
    data = h5.File(args.ipt, "r")
    geo = h5.File(args.geo, "r")
    drawer = Drawer(data, geo)

    # 画出分页的 PDF
    with PdfPages(args.opt) as pp:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_vertices_density(fig, ax)
        pp.savefig(figure=fig)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_pe_hit_time(fig, ax)
        pp.savefig(figure=fig)

        # Probe 函数图像使用极坐标绘制，注意 x 轴是 theta，y 轴是 r
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
        drawer.draw_probe(fig, ax)
        pp.savefig(figure=fig)
