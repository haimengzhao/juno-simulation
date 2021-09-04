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
    xs = (np.zeros((PRECISION, PRECISION))).flatten()
    ys = (np.sin(thetas) * ros).flatten()
    zs = (np.cos(thetas) * ros).flatten()

    # 多线程
    prob_t, prob_r, mean_t, mean_r, std_t, std_r = np.zeros((6, PRECISION, PRECISION))
    # 多线程
    pool = multiprocessing.Pool(processes=5)
    # 进度条
    pbar = tqdm(total=PRECISION*PRECISION)

    # 模拟光线
    res = np.array([pool.apply_async(gen_data, args=(xs[t], ys[t], zs[t], 0, 0), callback=lambda *x: pbar.update()) for t in range(PRECISION*PRECISION)])
    
    for i in range(PRECISION):
        for j in range(PRECISION):
            t = res[i*PRECISION+j].get()
            prob_t[i, j], prob_r[i, j], mean_t[i, j], mean_r[i, j], std_t[i, j], std_r[i, j] = t

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