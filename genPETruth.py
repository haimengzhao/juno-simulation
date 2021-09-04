from time import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.interpolate import interp2d
import utils
from get_prob_time import gen_data, get_prob_time
'''
gen_PETruth.py: 根据光学部分，ParticleTruth和PETruth，得到PETruth
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

PRECISION = 100

def gen_interp():
    
    # 插值用网格
    ro = np.linspace(0.2, 17.7, PRECISION)
    theta = np.linspace(0, 2*np.pi, PRECISION, endpoint=False)
    ros, thetas = np.meshgrid(ro, theta)
    # 测试点: yz平面
    xs = np.zeros((PRECISION, PRECISION))
    ys = np.sin(thetas) * ros
    zs = np.cos(thetas) * ros

    # 多线程
    prob_t, prob_r, mean_t, mean_r, std_t, std_r = np.zeros((6, PRECISION, PRECISION))
    bar = tqdm(total=PRECISION**2)
    for i in range(PRECISION):
        for j in range(PRECISION):
            res = gen_data(xs[i, j], ys[i, j], zs[i, j], 0, 0)
            prob_t[i, j], prob_r[i, j], mean_t[i, j], mean_r[i, j], std_t[i, j], std_r[i, j] = res
            bar.update()

    # 插值函数
    get_prob_t = interp2d(ro, theta, prob_t)
    get_prob_r = interp2d(ro, theta, prob_r)
    get_mean_t = interp2d(ro, theta, mean_t)
    get_mean_r = interp2d(ro, theta, mean_r)
    get_std_t = interp2d(ro, theta, std_t)
    get_std_r = interp2d(ro, theta, std_r)

    return get_prob_t, get_prob_r, get_mean_t, get_mean_r, get_std_t, get_std_r

gpt, gpr = gen_interp()[:2]

def allprob(r, theta):
    return gpt(r, theta) + gpr(r, theta)