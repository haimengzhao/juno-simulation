import numpy as np
import scipy.constants
import h5py
import numexpr as ne

'''
utility functions

xyz_from_spher:
transform spherical coordinates to cartesian coordinates

polar_from xyz:
transform cartesian coordinates to polar coordinates
given axis (theta0, phi0) and rotational invariance about the axis

save_file:
save data file

'''

# 常数定义
Ri = 17.71 # 液闪半径，单位m
Ro = 19.5 # PMT球心构成的球半径，单位m
n_water = 1.33 # 水的折射率
n_LS = 1.48 # 液闪的折射率
n_glass = 1.5 # 玻璃的折射率
c = scipy.constants.c # 光速，单位m/s
r_PMT = 0.508/2 # PMT的半径，单位m


def xyz_from_spher(r, theta, phi):
    '''
    transform spherical coordinates to cartesian coordinates

    input: r, theta, phi
    output: x, y, z
    '''
    ne.set_num_threads(16)

    x = ne.evaluate('r * sin(theta) * cos(phi)')
    y = ne.evaluate('r * sin(theta) * sin(phi)')
    z = ne.evaluate('r * cos(theta)')
    return x, y, z

def polar_from_xyz(R, theta0, phi0, x, y, z):
    '''
    transform cartesian coordinates to polar coordinates
    given axis (theta0, phi0) and rotational invariance about the axis

    input: R, theta0, phi0, x, y, z
    output: theta, r
    '''
    ne.set_num_threads(16)

    r = ne.evaluate('sqrt(x ** 2 + y ** 2 + z ** 2)')
    x0, y0, z0 = xyz_from_spher(R, theta0, phi0)
    distance = ne.evaluate('sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)')

    # cosine law
    theta = ne.evaluate('arccos((R ** 2 + r ** 2 - distance ** 2) / (2 * R * r))')

    return theta, r

def save_file(filename, ParticleTruth, PETruth, Waveform=None):
    '''
    save data file

    input: 
    filename, path of output file;
    ParticleTruth, PETruth, Waveform, structured arrays;
    '''
    with h5py.File(filename, "w") as opt:
        opt['ParticleTruth'] = ParticleTruth
        opt['PETruth'] = PETruth
        if Waveform != None:
            opt['Waveform'] = Waveform
