import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from time import time
from . import genPETruth

'''
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

precision = 500

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
    pool = multiprocessing.Pool()
    # 进度条

    # 模拟光线
    gpt, gpr = genPETruth.gen_interp()[:2]
    def allgpt(r, t):
        t = (t<np.pi)*t + (1-(t<np.pi))*(2*np.pi-t)
        return gpt(r, t) + gpr(r, t)
    res = allgpt(ros, thetas)

    res = np.clip(res.reshape(precision, precision), 5e-6, np.inf)
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