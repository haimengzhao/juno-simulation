'''
draw.py: 根据模拟数据绘图, 保存在pdf文件中

参数:
ipt: Input simulation data
-g, --geo: Geometry file
-o, --output: Output file

输出格式: 
pdf, 包含三张图片, 一页一张, 分别是:
顶点体密度随半径分布图;
PETime直方图;
Probe函数热力图.
'''

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp2d
from tqdm import tqdm
from scripts.utils import polar_from_xyz

# constants
Ri = 17.71e3 # inner radius / mm
Ro = 19.5e3 # outer radius / mm
Volume_i = 4 / 3 * np.pi * Ri ** 3 # volume of LS

# parameters
NumBins_Density = 30 # num of histogram bins when drawing density
NumBins_PETime = 1000 # num of histogram bins when drawing PETime
NumBins_Probe = 100 # num of histogram bins when drawing Prob

# 该类在测试时会用到，请不要私自修改函数签名，后果自负
class Drawer:
    def __init__(self, data, geo):
        self.simtruth = data["ParticleTruth"]
        self.petruth = data["PETruth"]
        self.geo = geo["Geometry"]

        self.N_vertices = len(data['ParticleTruth']) # total num of vertices
        self.rho0 = self.N_vertices / Volume_i # average density / mm^-3

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

        # extract histogram statistics
        n, bins, patches = ax.hist(r, bins=NumBins_Density, density=False)
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
                *map(
                    float,
                    ('%.2e'%self.rho0).split('e')
                )
            )
        )
        ax.set_ylim(0, 2 * self.rho0)
        ax.set_yticks(np.linspace(0, 2 * self.rho0, 5))
        ax.set_yticklabels(['%.1f' % (2 * i / 4) for i in range(5)])

        # density = dN / dV
        # dV = 4/3*pi*d(r^3)
        deltaVs = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        avgbins = (bins[:-1] + bins[1:]) / 2
        ax.scatter(avgbins, n / deltaVs, color='red')
        ax.plot(avgbins, n / deltaVs, color='red')
        # reference line
        ax.hlines(self.rho0, 0, Ri, color='black', linestyle='dashed')


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
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax.hist(time, bins=NumBins_PETime, density=False)

    def get_event_polar(self):
        '''
        ABANDONED
        originally used to plot probe
        get the polar distributions of events
        '''
        print('Preparing Event Polar Distribution')
        event_polar = np.zeros((17612 * self.N_vertices, 2))
        pmt_geo = self.geo[:17612]
        for event in tqdm(range(self.N_vertices)):
            x = self.simtruth['x'][event]
            y = self.simtruth['y'][event]
            z = self.simtruth['z'][event]
            theta, r = polar_from_xyz(
                Ro,
                pmt_geo['theta'] / 180 * np.pi,
                pmt_geo['phi']/ 180 * np.pi,
                x,
                y,
                z
            )
            event_polar[event:(event+17612), 0] = theta
            event_polar[event:(event+17612), 1] = r
        return event_polar


    def draw_probe(self, fig, ax):
        '''
        draw probe function
        average over all PMTs (Channels)
        probe = probe(theta, r)
        '''

        Events, Events_i = np.unique(self.petruth['EventID'],
                                     return_inverse=True)
        Channels, Channels_i = np.unique(self.petruth['ChannelID'],
                                         return_inverse=True)

        print('Replacing Event&Channel with xyz&geo')
        # replace ChannelID with corresponding geo
        geo_Channels_i = np.array(
            [np.where(self.geo['ChannelID'] == a)[0][0] for a in Channels]
        )
        pet_geo_i = geo_Channels_i[Channels_i]
        pet_geo = np.stack(
            [
                self.geo['theta'][pet_geo_i] / 180 * np.pi,
                self.geo['phi'][pet_geo_i] / 180 * np.pi
            ],
            -1
        )

        # replace EventID with corresponding xyz
        xyz_Event_i = np.array(
            [np.where(self.simtruth['EventID'] == a)[0][0] for a in Events]
        )
        pet_xyz_i = xyz_Event_i[Events_i]
        pet_xyz = np.stack(
            [
                self.simtruth['x'][pet_xyz_i],
                self.simtruth['y'][pet_xyz_i],
                self.simtruth['z'][pet_xyz_i]
            ],
            -1
        )

        # raplace xyz, geo with polar coordinates
        pet_polar = np.stack(
            polar_from_xyz(
                Ro,
                pet_geo[:, 0],
                pet_geo[:, 1],
                pet_xyz[:, 0],
                pet_xyz[:, 1],
                pet_xyz[:, 2]
            ),
            -1
        )

        # event_polar = self.get_event_polar() # ABANDANED

        # num of PE
        N_pe = len(pet_polar)

        print('Histograming')
        # extract histogram statistics
        h, redges, tedges, im = ax.hist2d(
            pet_polar[:, 1],
            pet_polar[:, 0],
            NumBins_Probe,
            range=[[0, Ri], [1e-4, np.pi-1e-4]],
            density=True
        )
        # hevent = ax.hist2d(
        #     event_polar[:, 0],
        #     event_polar[:, 1],
        #     NumBins_Probe,
        #     range=[[0, np.pi], [0, Ri]],
        #     density=False
        # )[0]
        ax.cla()

        # expand theta from [0, pi] to [0, 2pi]
        redges, tedges = (redges[:-1] + redges[1:]) / 2, (tedges[:-1] + tedges[1:]) / 2
        tedges_double = np.hstack([tedges, tedges + np.pi])
        h_double = np.hstack([h, np.fliplr(h)])
        # hevent_double = np.hstack([hevent, np.fliplr(hevent)]) # ABANDANED

        ThetaMesh, RMesh = np.meshgrid(tedges_double, redges)

        # d#PE/d#Vertices = d#PE/dV * dV/d#Vertices
        #                 = d#PE/(dV rho0)
        #                 = d#PE/(2pi r sin(theta) dr dtheta rho0)
        Z = h_double / (2*np.pi*RMesh*np.abs(np.sin(ThetaMesh))*self.rho0) * N_pe / 17612 / 4000

        print('Interploting')
        # interplot
        Z_interp = interp2d(tedges_double, redges, Z)
        ThetaInterp = np.linspace(0, 2 * np.pi, 1000)
        RInterp = np.linspace(0, Ri, 1000)

        # plot heatmap
        ax.set_title(r'Heatmap of the Probe Function $Probe(R, \theta)$')

        print('Drawing Heatmap')
        # pcm = ax.pcolormesh(
        #     ThetaMesh, RMesh, h_double / hevent_double,
        #     shading='auto', cmap=cm.get_cmap('jet')
        # ) # ABANDANED
        pcm = ax.pcolormesh(
            ThetaInterp,
            RInterp,
            Z_interp(ThetaInterp, RInterp),
            shading='auto',
            norm=colors.LogNorm(vmin=1e-1, vmax=1e2),
            cmap=cm.get_cmap('jet')
        )

        fig.colorbar(pcm, label='Expected Number of PE per Vertex')


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
        print('Ploting Vertex Density')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_vertices_density(fig, ax)
        pp.savefig(figure=fig)

        print('Ploting PETime')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_pe_hit_time(fig, ax)
        pp.savefig(figure=fig)

        print('Ploting Probe')
        # Probe 函数图像使用极坐标绘制，注意 x 轴是 theta，y 轴是 r
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
        drawer.draw_probe(fig, ax)
        pp.savefig(figure=fig)
