import numpy as np
import get_prob_time as prob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from time import time
from tqdm import tqdm

'''
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

precision = 100

# 画图用网格
ro = np.linspace(0.2, 17.7, precision)
theta = np.linspace(0, 2*np.pi, precision)
ros, thetas = np.meshgrid(ro, theta)

# 测试点: yz平面
xs = (np.zeros((precision, precision))).flatten()
ys = (np.sin(thetas) * ro).flatten()
zs = (np.cos(thetas) * ro).flatten()



if __name__ == '__main__':
    ti = time()

    # 多线程
    pool = multiprocessing.Pool(processes=7)
    # 进度条
    pbar = tqdm(total=precision*precision)

    # 模拟光线
    res = np.array([pool.apply_async(prob.get_PE_probability, (xs[t], ys[t], zs[t], 0, 0), callback=lambda *x: pbar.update()) for t in range(precision*precision)])
    res = np.frompyfunc((lambda x: x.get()), 1, 1)(res)
    res = np.clip(res.reshape(precision, precision).astype(np.float128), 1e-7, np.inf)
    pool.close()
    pool.join()

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
    c = ax.pcolormesh(thetas, ros, res, shading='auto', norm=colors.LogNorm(), cmap=plt.get_cmap('jet'))
    fig.colorbar(c, ax=ax)
    fig.savefig("probe.pdf")

    # 计时
    to = time()
    print(f'all time = {to-ti}')