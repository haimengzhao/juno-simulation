import numpy as np
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
    costheta = ne.evaluate('(R ** 2 + r ** 2 - distance ** 2) / (2 * R * r)')
    theta = np.arccos(np.clip(costheta, -1, 1))

    return theta, r

def save_file(filename, ParticleTruth, PETruth, Waveform):
    '''
    save data file

    input: 
    filename, path of output file;
    ParticleTruth, PETruth, Waveform, structured arrays;
    '''
    with h5py.File(filename, "w") as opt:
        opt['ParticleTruth'] = ParticleTruth
        opt['PETruth'] = PETruth
        opt['Waveform'] = Waveform
