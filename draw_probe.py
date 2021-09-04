import numpy as np
import get_prob_time as prob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from time import time
from tqdm import tqdm
import genPETruth

'''
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

precision = 1000

# 画图用网格
ro = np.linspace(0.2, 17.7, precision)
theta = np.linspace(0, 2*np.pi, precision, endpoint=False)
thetas, ros = np.meshgrid(theta, ro)
ros = ros.flatten()
thetas = thetas.flatten()

# 测试点: yz平面
# xs = (np.zeros((precision, precision))).flatten()
# ys = (np.sin(thetas) * ros).flatten()
# zs = (np.cos(thetas) * ros).flatten()



if __name__ == '__main__':
    ti = time()

    # 多线程
    pool = multiprocessing.Pool(processes=16)
    # 进度条
    pbar = tqdm(total=precision*precision)

    # 模拟光线
    res = np.array([pool.apply_async(genPETruth.allprob, args=(ros[t], thetas[t]), callback=lambda *x: pbar.update()) for t in range(precision*precision)])
    res = np.frompyfunc((lambda x: x.get()), 1, 1)(res)
    res = np.clip(res.reshape(precision, precision).astype(np.float64), 5e-5, np.inf)
    pool.close()
    pool.join()

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
    c = ax.pcolormesh(theta, ro, res, shading='auto', norm=colors.LogNorm(), cmap=plt.get_cmap('jet'))
    fig.colorbar(c, ax=ax)
    fig.savefig("probe.pdf")

    # 计时
    to = time()
    print(f'all time = {to-ti}')