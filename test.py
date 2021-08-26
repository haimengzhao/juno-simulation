import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from draw import Drawer as StudentDrawer


def __thetas(xs, ys, zs, pmt_ids, pmt_poss):
    vertex_poss = np.array([xs, ys, zs]).T
    vertex_poss_norm = np.linalg.norm(vertex_poss, axis=1)
    vertex_poss_norm = vertex_poss_norm.reshape(len(vertex_poss_norm), 1)
    vertex_poss /= vertex_poss_norm
    pmt_pos_by_ids = pmt_poss[pmt_ids]
    pmt_pos_by_ids_norm = np.linalg.norm(pmt_pos_by_ids, axis=1)
    pmt_pos_by_ids_norm = pmt_pos_by_ids_norm.reshape(len(pmt_pos_by_ids_norm), 1)
    pmt_pos_by_ids /= pmt_pos_by_ids_norm
    thetas = np.arccos(
        np.clip(np.einsum("ij, ij -> i", vertex_poss, pmt_pos_by_ids), -1, 1)
    )
    return thetas


npmt = 17612
R = 17710

neighborhood_r = 0.05 * R


def get_vertices(simtruth, geo_card):
    xs = simtruth["x"]
    ys = simtruth["y"]
    zs = simtruth["z"]
    xsr = xs.repeat(npmt)
    ysr = ys.repeat(npmt)
    zsr = zs.repeat(npmt)
    pmt_ids = np.tile(np.arange(npmt), len(xs))
    thetas = __thetas(xsr, ysr, zsr, pmt_ids, geo_card)
    rsr = np.sqrt(xsr ** 2 + ysr ** 2 + zsr ** 2)
    return rsr, thetas


def get_pes(simtruth, petruth, geo_card):
    simtruth = pd.DataFrame(simtruth, columns=["EventID", "x", "y", "z", "p"])
    petruth = pd.DataFrame(petruth, columns=["EventID", "ChannelID", "PETime"])
    pes = pd.merge(simtruth, petruth, on=["EventID"], how="outer")
    xs = np.array(pes["x"])
    ys = np.array(pes["y"])
    zs = np.array(pes["z"])
    pmt_ids = np.array(pes["ChannelID"])
    thetas = __thetas(xs, ys, zs, pmt_ids, geo_card)
    rs = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    ts = np.array(pes["PETime"])
    return rs, thetas, ts


class Drawer:
    def __init__(self, data, geo):
        self.simtruth = data["ParticleTruth"][()]
        self.petruth = data["PETruth"][()]
        self.waveform = None
        if "Waveform" in data:
            self.waveform = data["Waveform"]
        geo = geo["Geometry"][:npmt]
        geo["theta"] /= 180 / np.pi
        geo["phi"] /= 180 / np.pi
        self.geo_card = np.zeros((len(geo), 3))
        self.geo_card[:, 0] = np.sin(geo["theta"]) * np.cos(geo["phi"])
        self.geo_card[:, 1] = np.sin(geo["theta"]) * np.sin(geo["phi"])
        self.geo_card[:, 2] = np.cos(geo["theta"])
        self.v_rs, self.v_thetas = get_vertices(self.simtruth, self.geo_card)
        self.pe_rs, self.pe_thetas, self.pe_ts = get_pes(
            self.simtruth, self.petruth, self.geo_card
        )

    def draw_vertices_density(self, fig, ax):
        ax.hist(self.v_rs, range=(0, R), bins=100, weights=1 / (self.v_rs ** 2))

    def draw_pe_hit_time(self, fig, ax):
        ax.hist(self.petruth["PETime"], bins=100)

    def draw_pe_hit_time_rth(self, fig, ax, r, theta):
        sts = self.pe_ts[
            self.pe_rs ** 2
            + r ** 2
            - 2 * self.pe_rs * r * np.cos(self.pe_thetas - theta)
            <= neighborhood_r ** 2
        ]
        ax.hist(sts, bins=100)

    def draw_probe(self, fig, ax):
        f_v_rs = np.hstack([self.v_rs, self.v_rs])
        f_v_thetas = np.hstack([self.v_thetas, 2 * np.pi - self.v_thetas])
        f_pe_rs = np.hstack([self.pe_rs, self.pe_rs])
        f_pe_thetas = np.hstack([self.pe_thetas, 2 * np.pi - self.pe_thetas])
        r_bins = np.linspace(0, R, 51)
        theta_bins = np.linspace(0, 2 * np.pi, 201)
        binning = [r_bins, theta_bins]
        hist_pe, binr, bintheta = np.histogram2d(f_pe_rs, f_pe_thetas, bins=binning)
        hist_predict, _, _ = np.histogram2d(f_v_rs, f_v_thetas, bins=binning)
        X, Y = np.meshgrid(bintheta, binr)
        cm = ax.pcolormesh(X, Y, hist_pe / hist_predict, norm=LogNorm(), cmap="jet")
        fig.colorbar(cm)

    def draw_waveform(self, fig, ax):
        wave = self.waveform[-1]
        ax.plot(wave["Waveform"])
        pes = self.petruth[
            np.logical_and(
                self.petruth["EventID"] == wave["EventID"],
                self.petruth["ChannelID"] == wave["ChannelID"],
            )
        ]
        scale = np.max(wave["Waveform"]) - np.min(wave["Waveform"])
        ax.vlines(
            pes["PETime"],
            np.min(wave["Waveform"]) - scale / 3,
            np.max(wave["Waveform"]) + scale / 3,
        )


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ipt", type=str, help="Input simulation data")
parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
args = parser.parse_args()

data = h5.File(args.ipt, "r")
geo = h5.File(args.geo, "r")
drawer = Drawer(data, geo)
sdrawer = StudentDrawer(data, geo)

with PdfPages(args.opt) as pp:
    figsize = (12.8, 4.8)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    drawer.draw_vertices_density(fig, ax)
    ax = fig.add_subplot(1, 2, 2)
    sdrawer.draw_vertices_density(fig, ax)
    pp.savefig(figure=fig)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    drawer.draw_pe_hit_time(fig, ax)
    ax = fig.add_subplot(1, 2, 2)
    sdrawer.draw_pe_hit_time(fig, ax)
    pp.savefig(figure=fig)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    drawer.draw_pe_hit_time_rth(fig, ax, 0.95 * R, 0)
    ax = fig.add_subplot(1, 2, 2)
    drawer.draw_pe_hit_time_rth(fig, ax, 0.95 * R, np.pi)
    pp.savefig(figure=fig)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1, projection="polar", theta_offset=np.pi / 2)
    drawer.draw_probe(fig, ax)
    ax = fig.add_subplot(1, 2, 2, projection="polar", theta_offset=np.pi / 2)
    sdrawer.draw_probe(fig, ax)
    pp.savefig(figure=fig)

    if "Waveform" in data.keys():
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        drawer.draw_waveform(fig, ax)
        pp.savefig(figure=fig)
