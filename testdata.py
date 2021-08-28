import numpy as np
import h5py

# constants
Ri = 17.71e3 # inner radius / mm
Ro = 19.5e3 # outer radius / mm

# testdata generator
if __name__ == '__main__':
    N = 4000
    N_photon_per_vertex = 10

    # x = np.random.normal(loc=Ri/2, scale=Ri/8, size=N)
    x = np.random.uniform(0, Ri, N)
    y = x
    z = x

    # PET_EventID = np.array(range(0, N * N_photon_per_vertex))
    # PET_ChannelID = np.zeros(N * N_photon_per_vertex)
    # PETime = np.random.uniform(100, 500, N * N_photon_per_vertex)

    PET_EventID = np.array([0, 1, 0, 0, 0, 1, 1])
    PET_ChannelID = np.array([0, 0, 1, 1, 2, 2, 2])
    PETime = np.array([100, 200, 300, 300, 500, 600, 600])


    with h5py.File('./data.h5', mode='w') as file:
        file.create_group('ParticleTruth')
        file.create_group('PETruth')
        file['ParticleTruth']['EventID'] = np.array(range(0, N))
        file['ParticleTruth']['x'] = x
        file['ParticleTruth']['y'] = y
        file['ParticleTruth']['z'] = z

        file['PETruth']['EventID'] = PET_EventID
        file['PETruth']['ChannelID'] = PET_ChannelID
        file['PETruth']['PETime'] = PETime
