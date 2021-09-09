'''
drawProbe.py: legacy架构下绘制probe图像

警告: 这是legacy架构的函数, 在本架构下已没有意义. 但由于报告中有legacy画的图,
      为了报告的可复现性, 修复了其正常功能, 以供参考.

描述: 根据模拟时使用的get_PE_probability函数绘制probe图像
      可与draw.py中根据data.h5绘制的probe图像进行对比
'''

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing
from . import genPETruth

# 取PRECISION*PRECISION个点画图
PRECISION = 500

# 画图用网格
ro = np.linspace(0.2, 17.7, PRECISION)
theta = np.linspace(0, 2*np.pi, PRECISION, endpoint=False)
thetas, ros = np.meshgrid(theta, ro)
ros = ros.flatten()
thetas = thetas.flatten()

# 测试点: yz平面
# xs = (np.zeros((PRECISION, PRECISION))).flatten()
# ys = (np.sin(thetas) * ros).flatten()
# zs = (np.cos(thetas) * ros).flatten()


if __name__ == '__main__':
    ti = time()

    # 多线程
    pool = multiprocessing.Pool()

    # 模拟光线, gpt为透射概率的插值函数，gpr为反射概率
    gpt, gpr = genPETruth.gen_interp()[:2]
    def allgpt(r, t):
        t = (t<np.pi)*t + (1-(t<np.pi))*(2*np.pi-t)
        return gpt(r, t) + gpr(r, t)
    res = allgpt(ros, thetas)

    res = np.clip(res.reshape(PRECISION, precision), 5e-6, np.inf)
    pool.close()
    pool.join()

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar", theta_offset=np.pi / 2)
    c = ax.pcolormesh(
        theta,
        ro,
        res,
        shading='auto',
        norm=colors.LogNorm(),
        cmap=plt.get_cmap('jet')
    )
    fig.colorbar(c, ax=ax)
    fig.savefig("probe.pdf")

    # 计时
    to = time()
    print(f'all time = {to-ti}')
