import numpy as np
import get_prob_time as prob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import ctypes

ro = np.linspace(0.2, 17.7, 30)
theta = np.linspace(0, 2*np.pi, 30)
ros, thetas = np.meshgrid(ro, theta)

xs = (np.zeros((30, 30))).flatten()
ys = (np.cos(thetas) * ro).flatten()
zs = (np.sin(thetas) * ro).flatten()




# a = np.zeros((30, 30), dtype=np.float128)
# shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
# shm = multiprocessing.Array(ctypes.c_double, 30*30, lock=False)
# # res = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
# res = np.frombuffer(shm, dtype=ctypes.c_double).reshape(30, 30)
# def task(x, i, j, name):
#     shm = shared_memory.SharedMemory(name=name)
#     res = np.ndarray((30, 30), dtype=np.float128, buffer=shm.buf)
#     res[i, j] = x



if __name__ == '__main__':
    # smm = SharedMemoryManager()
    # smm.start()
    # s = np.zeros((30, 30), dtype=np.float128)
    # sm = smm.SharedMemory(s.nbytes)
    # res = np.ndarray((30, 30), dtype=np.float128, buffer=sm.buf)
    pool = multiprocessing.Pool(processes=8)
    # for i in range(30):
    #     for j in range(30):
    #         pool.apply_async(prob.get_PE_probability, args=(xs[i, j], ys[i, j], zs[i, j], 0, 0), callback=(lambda x: task(x, i, j, sm.buf)))
    

    res = np.array([pool.apply_async(prob.get_PE_probability, (xs[t], ys[t], zs[t], 0, 0)) for t in range(30*30)])
    res = np.frompyfunc((lambda x: x.get()), 1, 1)(res)
    res = np.clip(res.reshape(30, 30).astype(np.float128), 1e-7, np.inf)
    pool.close()
    pool.join()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    c = ax.pcolormesh(thetas, ros, res, shading='auto', norm=colors.LogNorm())
    fig.colorbar(c, ax=ax)
    fig.savefig("probe.pdf")
