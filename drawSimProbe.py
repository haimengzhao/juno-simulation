'''
drawSimProbe.py: 使用double-search法绘制probe图像

描述: double-search算法的画图脚本，使用genSimProbe的插值函数绘制probe图像
      可与draw.py中根据data.h5绘制的probe图像进行对比
      绘制的图像名为probe.pdf
'''

from time import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scripts.genSimProbe import gen_interp

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

    # 模拟光线, gpt为光子打到PMT上属于透射事件概率的插值函数，
    #           gpr为光子打到PMT上属于先反射后透射事件概率的插值函数
    gpt, gpr = gen_interp()[:2]

    def allgpt(r, t):
        '''
        内部函数，得到(r, theta)位置处的总概率，即透射概率+反射概率
        '''
        t = (t < np.pi)*t + (1-(t < np.pi))*(2*np.pi-t)
        return gpt(r, t) + gpr(r, t)
    print("正在生成画图用点...")
    res = allgpt(ros, thetas)

    res = np.clip(res.reshape(PRECISION, PRECISION), 5e-6, np.inf)
    pool.close()
    pool.join()

    # 画图
    print("正在画图...")
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
