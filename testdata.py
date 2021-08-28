import numpy as np
import h5py

# constants
Ri = 17.71e3 # inner radius / mm
Ro = 19.5e3 # outer radius / mm

# testdata generator
if __name__ == '__main__':
    N = 4000
    N_photon_per_vertex = 1

    # x = np.random.normal(loc=Ri/2, scale=Ri/8, size=N)
    EventID = np.array(range(0, N))
    x = np.random.uniform(0, Ri, N)
    y = x
    z = x
    p = np.ones_like(x)
    PT = np.array(list(zip(EventID, x, y, z, p)), dtype=
        [('EventID', 'i4'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('p', 'f8')])

    PET_EventID = np.array(range(0, N * N_photon_per_vertex))
    PET_ChannelID = np.random.uniform(0, 1000, N * N_photon_per_vertex)
    PETime = np.random.uniform(100, 500, N * N_photon_per_vertex)

    # PET_EventID = np.array([0, 1, 0, 0, 0, 1, 1])
    # PET_ChannelID = np.array([0, 0, 1, 1, 2, 2, 2])
    # PETime = np.array([100, 200, 300, 300, 500, 600, 600])
    PET = np.array(list(zip(PET_EventID, PET_ChannelID, PETime)), dtype=
        [('EventID', 'i4'), ('ChannelID', 'i4'), ('PETime', 'f8')])


    with h5py.File('./data.h5', mode='w') as file:
        file['ParticleTruth'] = PT
        file['PETruth'] = PET