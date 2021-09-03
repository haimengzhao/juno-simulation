from time import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.interpolate import interp2d
import utils
from get_prob_time import gen_data
'''
gen_PETruth.py: 根据光学部分，ParticleTruth和PETruth，得到PETruth
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

PRECISION = 100

def gen_interp():
    
    # 插值用网格
    ro = np.linspace(0.2, 17.7, PRECISION)
    theta = np.linspace(0, 2*np.pi, PRECISION)
    ros, thetas = np.meshgrid(ro, theta)
    # 测试点: yz平面
    xs = (np.zeros((PRECISION, PRECISION))).flatten()
    ys = (np.sin(thetas) * ro).flatten()
    zs = (np.cos(thetas) * ro).flatten()
    
    def gen(x):
        return gen_data(x[0],x[1],x[2],x[3])

    class dots:  
            def __init__(self, count, xs, ys, zs):  
                self.count = count  
                self.xs = xs[:]  
                self.ys = ys[:]  
                self.zs = zs[:]  
            def __iter__(self):  
                self.t = 0  
                self.dot = (self.xs[self.t], self.ys[self.t], self.zs[self.t], 0, 0)  
                return self  
            def __next__(self): 
                dot = self.dot  
                if self.t > self.count-1:
                    raise StopIteration 
                self.t += 1 
                if self.t <= self.count-1: 
                    self.dot = (self.xs[self.t], self.ys[self.t], self.zs[self.t], 0, 0)  
                return dot
    # 多线程
    with multiprocessing.Pool(processes=8) as p:

        # 计算网格对应的PE_probability
        prob_t, prob_r, mean_t, mean_r, std_t, std_r = np.array(
            list(tqdm(
                p.imap(
                    gen,
                    dots(PRECISION**2, xs, ys, zs)
                ),
                total = PRECISION**2
            ))
        )

    prob_t = np.frompyfunc((lambda x: x.get()), 1, 1)(prob_t)
    prob_t = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    prob_r = np.frompyfunc((lambda x: x.get()), 1, 1)(prob_r)
    prob_r = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    mean_t = np.frompyfunc((lambda x: x.get()), 1, 1)(mean_t)
    mean_t = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    mean_r = np.frompyfunc((lambda x: x.get()), 1, 1)(mean_r)
    mean_r = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    std_t = np.frompyfunc((lambda x: x.get()), 1, 1)(std_t)
    std_t = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    std_r = np.frompyfunc((lambda x: x.get()), 1, 1)(std_t)
    std_r = np.clip(res.reshape(PRECISION, PRECISION), 5e-5, np.inf)
    pool.close()
    pool.join()

    # 插值函数
    get_prob_t = interp2d(ro, theta, prob_t)
    get_prob_r = interp2d(ro, theta, prob_r)
    get_mean_t = interp2d(ro, theta, mean_t)
    get_mean_r = interp2d(ro, theta, mean_r)
    get_std_t = interp2d(ro, theta, std_t)
    get_std_r = interp2d(ro, theta, std_r)

    return get_prob_t, get_prob_r, get_mean_t, get_mean_r, get_std_t, get_srd_r
