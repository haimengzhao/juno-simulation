'''
event.py: 顶点与光子模拟

主要接口：generate_events
生成球内均匀分布的顶点位置坐标，使用非齐次泊松分布采样光子时间
'''

import multiprocessing
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm
from numba import njit
from . import utils

LS_RADIUS = utils.Ri # 液闪的半径，单位m
SIGMA = 5 # 正态分布的标准差
TAU = 20 # 指数衰减函数e^(-t/tau)中的tau
NORM_FACTOR = 69.1504473757916# 期望的归一化系数
T_MAX = 500 # 只考虑500ns以内产生的光子
PRECISION = 1000 # expectation取样时的间隔为其倒数

def expectation(t):
    '''
    README中的lambda(t)，非齐次泊松过程的期望
    输入：时间t
    输出：期望值
    '''
    return (quad(lambda s: np.exp(-s-(t/TAU-s)**2/2/SIGMA/SIGMA),
                 0, np.inf)[0]
            * NORM_FACTOR)

def generate_events(number_of_events):
    '''
    描述：生成事例
    输入：number_of_events: event数量
    输出：ParticleTruth，PhotonTruth两个结构化数组
          ParticleTruth形状为(number_of_events, 5)，具有字段：
            EventID: 事件编号        '<i4'
            x:       顶点坐标x/mm    '<f8'
            y:       顶点坐标y/mm    '<f8'
            z:       顶点坐标z/mm    '<f8'
            p:       顶点动量/MeV    '<f8'
          Photons形状为(不定, 3)，具有字段:
            EventID:  事件编号                       '<i4'
            PhotonID: 光子编号（每个事件单独编号）   '<i4'
            GenTime:  从顶点产生到光子产生的时间/ns  '<f8'
    算法描述：
    1. 生成顶点坐标(x, y, z)
       方法：生成球坐标，r用一个与r^2成正比的采样函数
                         theta和phi均匀分布
             转为xyz
    2. 生成光子数目与GenTime
       方法：先算出卷积后的lambda(t), 得到其最大值lambda*
             定义截止时间，将截止时间内产生的光子作为总共的光子
             用lambda*的齐次泊松分布模拟截止时间内的光子事件
             筛选事件，有lambda(t)/lambda*的可能性事件留下
    3. 转化为输出格式输出
    '''

    # 初始化expectation
    print("初始化expectation...")
    step = 1/PRECISION
    expect_list = np.zeros(int(T_MAX/step) + 1)
    with multiprocessing.Pool(8) as p:
        expect_list = np.array(
            list(tqdm(
                p.imap(
                    expectation,
                    np.arange(0, T_MAX+step, step)
                ),
                total=expect_list.shape[0]
            ))
        )
    max_expect = max(expect_list)
    # 线性插值
    @njit
    def linear_intp(t, expect_list, PRECISION):
        floor = np.floor(t*PRECISION).astype(np.int64)
        return (((t*PRECISION) - floor) * expect_list[floor] +
                (-(t*PRECISION) + floor + 1) * expect_list[floor+1])

    # 初始化rng
    rng = np.random.default_rng()

    print("正在生成event位置...")
    # 生成event的球坐标位置
    event_r = rng.power(3, size=number_of_events) * LS_RADIUS
    event_theta = rng.random(size=number_of_events) * np.pi
    event_phi = rng.random(size=number_of_events) * np.pi * 2

    # 转为直角坐标
    # event_coordinates形状为(number_of_events, 3)
    # event_coordinates[EVENT_ID][0]表示第EVENT_ID的x坐标
    event_coordinates = np.array(
        utils.xyz_from_spher(event_r, event_theta, event_phi)
        ).transpose()

    # 生成ParticleTruth
    par_tr_dtype = [
        ('EventID', '<i4'),
        ('x', '<f8'),
        ('y', '<f8'),
        ('z', '<f8'),
        ('p', '<f8')
    ]
    Particle_Truth = np.zeros(number_of_events, dtype=par_tr_dtype)
    Particle_Truth['EventID'] = np.arange(number_of_events)
    Particle_Truth['x'] = event_coordinates[:, 0]*1000
    Particle_Truth['y'] = event_coordinates[:, 1]*1000
    Particle_Truth['z'] = event_coordinates[:, 2]*1000
    Particle_Truth['p'] = np.ones(number_of_events)
    print("ParticleTruth表生成完成！")

    # 生成光子，先用齐次泊松分布，再用expectation来thin
    print("生成光子中...")
    photon_counts = np.round(
        rng.poisson(max_expect*T_MAX, number_of_events)
        ).astype(int)

    event_ids = np.zeros(sum(photon_counts)) #这是可能的最大shape
    photon_ids = np.zeros(sum(photon_counts))
    gen_times = np.zeros(sum(photon_counts))
    start = 0

    for event_id, photon_count in enumerate(tqdm(photon_counts)):
        gen_time = np.sort(rng.random(photon_count) * T_MAX)
        gen_time = gen_time[
            rng.random() < linear_intp(gen_time, expect_list, PRECISION)/max_expect
        ]
        real_photon_count = gen_time.shape[0]
        event_ids[start:(start + real_photon_count)] = event_id
        photon_ids[start:(start + real_photon_count)] = np.arange(real_photon_count)
        gen_times[start:(start + real_photon_count)] = gen_time
        start = start + real_photon_count

    # 生成PhotonTruth
    pho_tr_dtype = [
        ('EventID', '<i4'),
        ('PhotonID', '<i4'),
        ('GenTime', '<f8')
    ]
    Photon_Truth = np.zeros(start, dtype=pho_tr_dtype)
    Photon_Truth['EventID'] = event_ids[:start]
    Photon_Truth['PhotonID'] = photon_ids[:start]
    Photon_Truth['GenTime'] = gen_times[:start]
    print("PhotonTruth表生成完成！")

    return Particle_Truth, Photon_Truth
