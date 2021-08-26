import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# 该类在测试时会用到，请不要私自修改函数签名，后果自负
class Drawer:
    def __init__(self, data, geo):
        self.simtruth = data["ParticleTruth"]
        self.petruth = data["PETruth"]
        self.geo = geo["Geometry"]

    def draw_vertices_density(self, fig, ax):
        print("TODO: Draw vertices density by radius")

    def draw_pe_hit_time(self, fig, ax):
        print("TODO: Draw histogram of PE hit time")

    def draw_probe(self, fig, ax):
        print("TODO: Draw probe R(r, theta)")


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
