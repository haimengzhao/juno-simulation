import numpy as np

'''
utility functions
'''

def xyz_from_spher(r, theta, phi):
    '''
    transform spherical coordinates to cartesian coordinates

    input: r, theta, phi
    output: x, y, z
    '''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def polar_from_xyz(R, theta0, phi0, x, y, z):
    '''
    transform cartesian coordinates to polar coordinates
    given axis (theta0, phi0) and rotational invariance about the axis

    input: R, theta0, phi0, x, y, z
    output: theta, r
    '''
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x0, y0, z0 = xyz_from_spher(R, theta0, phi0)
    distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    # cosine law
    theta = np.arccos((R ** 2 + r ** 2 - distance ** 2) / (2 * R * r))

    return theta, r