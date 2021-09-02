import numpy as np
import get_prob_time as prob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from time import time

ro = np.linspace(0.2, 17.7, 100)
theta = np.linspace(0, 2*np.pi, 100)
ros, thetas = np.meshgrid(ro, theta)

xs = (np.zeros((100, 100))).flatten()
ys = (np.cos(thetas) * ro).flatten()
zs = (np.sin(thetas) * ro).flatten()



if __name__ == '__main__':
    ti = time()

    pool = multiprocessing.Pool(processes=8)
    res = np.array([pool.apply_async(prob.get_PE_probability, (xs[t], ys[t], zs[t], 0, 0)) for t in range(100*100)])
    res = np.frompyfunc((lambda x: x.get()), 1, 1)(res)
    res = np.clip(res.reshape(100, 100).astype(np.float128), 5e-5, np.inf)
    pool.close()
    pool.join()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    c = ax.pcolormesh(thetas, ros, res, shading='auto', norm=colors.LogNorm(), cmap=plt.get_cmap('jet'))
    fig.colorbar(c, ax=ax)
    fig.savefig("probe.pdf")

    to = time()
    print(f'all time = {to-ti}')