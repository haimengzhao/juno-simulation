from time import time
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.interpolate import RectBivariateSpline
import numexpr as ne
import utils
from get_prob_time import gen_data
'''
gen_PETruth.py: 根据光学部分，ParticleTruth和PETruth，得到PETruth
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''

PRECISION = 100
LS_RADIUS = 17.71
PMT_R = 19.5

def gen_interp():
    '''
    生成插值函数，使用其中的插值函数来近似get_PE_probability与get_random_PE_time
    '''
    print("正在生成插值函数...")

    # 插值用网格
    ro = np.linspace(0.2, LS_RADIUS, PRECISION)
    theta = np.linspace(0, np.pi, PRECISION)
    ros, thetas = np.meshgrid(ro, theta)

    # 测试点: yz平面
    xs = (np.zeros((PRECISION, PRECISION))).flatten()
    ys = (np.sin(thetas) * ros).flatten()
    zs = (np.cos(thetas) * ros).flatten()

    prob_t, prob_r, mean_t, mean_r, std_t, std_r = np.zeros((6, PRECISION, PRECISION))
   
    # 多线程
    pool = multiprocessing.Pool(processes=16)
    # 进度条
    pbar = tqdm(total=PRECISION*PRECISION)

    # 模拟光线
    res = np.array([pool.apply_async(gen_data, args=(xs[t], ys[t], zs[t], 0, 0), callback=lambda *x: pbar.update()) for t in range(PRECISION*PRECISION)])
    
    for i in range(PRECISION):
        for j in range(PRECISION):
            t = res[i*PRECISION+j].get()
            prob_t[i, j], prob_r[i, j], mean_t[i, j], mean_r[i, j], std_t[i, j], std_r[i, j] = t

    # 插值函数
    get_prob_t = RectBivariateSpline(ro, theta, prob_t, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    get_prob_r = RectBivariateSpline(ro, theta, prob_r, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    get_mean_t = RectBivariateSpline(ro, theta, mean_t, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    get_mean_r = RectBivariateSpline(ro, theta, mean_r, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    get_std_t = RectBivariateSpline(ro, theta, std_t, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    get_std_r = RectBivariateSpline(ro, theta, std_r, kx=1, ky=1, bbox=[0, 17.71, 0, np.pi]).ev
    
    print("插值函数生成完毕！")
    return get_prob_t, get_prob_r, get_mean_t, get_mean_r, get_std_t, get_std_r

def to_relative_position(x, y, z, PMT_phi, PMT_theta):
    '''
    将顶点位置x, y, z与PMT位置phi, theta转化成插值时的r, theta
    '''
    PMT_x, PMT_y, PMT_z = utils.xyz_from_spher(PMT_R, PMT_theta, PMT_phi)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = ne.evaluate("arccos((PMT_R**2 + r**2 - (PMT_x - x)**2 - (PMT_y - y)**2 - (PMT_z - z)**2 )/(2*PMT_R*r))")
    return r, theta

def get_PE_Truth(ParticleTruth, PhotonTruth, PMT_list, number_of_events):
    '''
    通过Particle_Truth与Photon_Truth，生成PE_Truth
    '''
    PMT_COUNT = PMT_list.shape[0]
    get_prob_t, get_prob_r, get_mean_t, get_mean_r, get_std_t, get_std_r = gen_interp()
    rng = np.random.default_rng()

    def intp_PE_probability(x, y, z, PMT_phi, PMT_theta):
        '''
        用于代替光学部分中的get_PE_probability，使用插值函数
        '''
        r, theta = to_relative_position(x, y, z, PMT_phi, PMT_theta)
        return get_prob_t(r, theta)[0], get_prob_r(r, theta)[0]

    def intp_random_PE_time(x, y, z, PMT_phi, PMT_theta, prob_t, prob_r):
        '''
        用于代替光学部分中的get_random_PE_time，使用插值函数
        '''
        r, theta = to_relative_position(x, y, z, PMT_phi, PMT_theta)
        if rng.random() < prob_t / (prob_t + prob_r):
            return rng.normal(
                get_mean_t(r, theta)[0],
                get_std_t(r, theta)[0]
            )
        else:
            return rng.normal(
                get_mean_r(r, theta)[0],
                get_std_r(r, theta)[0]
            )

    PE_prob_cumsum = np.zeros((number_of_events, PMT_COUNT))
    PE_prob_array = np.zeros((number_of_events, PMT_COUNT, 2))
    
    print("正在给每个event生成打到每个PMT上的概率...")
    for event in tqdm(ParticleTruth):
        for PMT in PMT_list:
            PE_prob_array[event['EventID']][PMT['ChannelID']][:] = intp_PE_probability(
                event['x']/1000, event['y']/1000, event['z']/1000,
                PMT['phi']/180*np.pi, PMT['theta']/180*np.pi
            )

        PE_prob_cumsum[event['EventID']][:] = np.cumsum(
            np.sum(PE_prob_array[event['EventID']], axis=1)
        )
    
    print("正在模拟每个光子打到的PMT与PETime...")
    PE_event_ids = np.zeros(PhotonTruth.shape[0])
    PE_channel_ids = np.zeros(PhotonTruth.shape[0])
    PE_petimes = np.zeros(PhotonTruth.shape[0])

    index = 0
    for photon in tqdm(PhotonTruth):
        channel_hit = np.asarray(rng.random() < PE_prob_cumsum[photon['EventID']][:]).nonzero()[0]
        if channel_hit.shape[0] > 0:
            PE_event_ids[index] = photon['EventID']
            PE_channel_ids[index] = channel_hit[0]
            PE_petimes[index] = photon['GenTime'] + intp_random_PE_time(
                ParticleTruth[photon['EventID']]['x']/1000,
                ParticleTruth[photon['EventID']]['y']/1000,
                ParticleTruth[photon['EventID']]['z']/1000,
                PMT_list[channel_hit[0]]['phi']/180*np.pi,
                PMT_list[channel_hit[0]]['theta']/180*np.pi,
                PE_prob_array[event['EventID']][channel_hit[0]][0],
                PE_prob_array[event['EventID']][channel_hit[0]][1]
            )
            index += 1

    print("正在生成PETruth表...")
    pe_tr_dtype = [
        ('EventID', '<i4'),
        ('ChannelID', '<i4'),
        ('PETime', '<f8')
    ]
    PETruth = np.zeros(index, dtype=pe_tr_dtype)
    PETruth['EventID'] = PE_event_ids[:index]
    PETruth['ChannelID'] = PE_channel_ids[:index]
    PETruth['PETime'] = PE_petimes[:index]
    
    print("PETruth表生成完毕！")
    return PETruth
