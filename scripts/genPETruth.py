import numpy as np
from tqdm import tqdm
import multiprocessing
import numexpr as ne
from numba import njit
from . import utils
'''
gen_PETruth.py: 根据光学部分，ParticleTruth和PETruth，得到PETruth
根据模拟时使用的get_PE_probability函数绘制probe图像
可与draw.py中根据data.h5绘制的probe图像进行对比.
'''
n_water = 1.33
n_LS = 1.48
n_glass = 1.5
eta = n_LS / n_water
Ri = 17.71
Ro = 19.5
r_PMT = 0.508/2
c = 3e8

PETruth = {}
PETruth['EventID'] = []
PETruth['ChannelID'] = []
PETruth['PETime'] = []
PMT_num = 17612

def transist(coordinates, velocities, times, events, can_reflect):
    # 求解折射点
    cv = np.einsum('kn, kn->n', coordinates, velocities)
    ts = -cv + np.sqrt(cv**2 - (np.einsum('kn, kn->n', coordinates, coordinates)-Ri**2))    #到达液闪边界的时间
    edge_points = coordinates + ts * velocities
    
    # 计算增加的时间
    new_times = times + (n_LS/c)*ts

    # 计算入射角，出射角
    normal_vectors = -edge_points / Ri
    incidence_vectors = velocities
    vertical_of_incidence = np.maximum(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1)
    incidence_angles = np.arccos(-vertical_of_incidence)
    
    # 判断全反射
    max_incidence_angle = np.arcsin(n_water/n_LS)
    can_transmit = (incidence_angles < max_incidence_angle)
    all_reflect = 1 - can_transmit

    #计算折射光，反射光矢量与位置
    reflected_velocities = velocities - 2 * vertical_of_incidence * normal_vectors

    delta = ne.evaluate('1 - eta**2 * (1 - vertical_of_incidence**2)')
    new_velocities = ne.evaluate('(eta*incidence_vectors - (eta*vertical_of_incidence + sqrt(abs(delta))) * normal_vectors) * can_transmit') #取绝对值避免出错

    # 计算折射系数
    emergence_angles = np.arccos(np.minimum(np.einsum('kn, kn->n', new_velocities, -normal_vectors), 1))
    Rs = ne.evaluate('(sin(emergence_angles - incidence_angles)/sin(emergence_angles + incidence_angles))**2')
    Rp = ne.evaluate('(tan(emergence_angles - incidence_angles)/tan(emergence_angles + incidence_angles))**2')
    R = (Rs+Rp)/2

    # 选出需要折射和反射的光子
    probs = np.random.random(times.shape[0])
    need_transmit = (probs>R) * can_transmit
    need_reflect = (1-need_transmit) * can_reflect

    # 处理需要折射出去的光子
    hit_PMT(coordinates[need_transmit], new_velocities[need_transmit], new_times[need_transmit], events[need_transmit], can_reflect[need_transmit])

    # 处理需要继续反射的光子
    transist(coordinates[need_reflect], reflected_velocities[need_reflect], new_times[need_reflect], events[need_reflect], np.zeros(need_reflect.shape[0]))



def go_inside(coordinates, velocities, times, events):
    pass



def distance(coordinates, velocities, PMT_coordinates):
    '''
    coordinates: (3, n)
    PMT_coordinates: (3, m)
    接收一族光线，给出其未来所有时间内与给定PMT的最近距离
    注意：光线是有方向的，如果光子将越离越远，那么将返回负数距离
    '''
    coordinates = coordinates.reshape(3, 1, coordinates.shape[1])
    PMT_coordinates = PMT_coordinates.reshape(3, PMT_coordinates.shape[1], 1)
    velocities = velocities.reshape(3, 1, velocities.shape[1])

    new_ts = np.einsum('kmn, kn->mn', PMT_coordinates - coordinates, velocities)
    nearest_points = coordinates + new_ts * velocities
    distances = np.linalg.norm(nearest_points - PMT_coordinates, axis=0) * np.sign(new_ts) 
    return distances


def hit_this_PMT(coordinates, velocities, times, events, can_reflect, PMT_coordinates):
    pass


def hit_PMT(coordinates, velocities, times, events, can_reflect, PMT_coordinates, fromthis=False):
    distances = distance(coordinates, velocities, PMT_coordinates)
    allowed = distances < r_PMT
    photon_num = times.shape[0]

    PMT2edge = coordinates.reshape(3, 1, photon_num) - PMT_coordinates.reshape(3, PMT_num, 1)
    ts = -np.einsum('kmn, kn->mn', PMT2edge, velocities) +\
          np.sqrt(np.einsum('kmn, kn->mn', PMT2edge, velocities)**2 - np.einsum('kmn, kmn->mn', PMT2edge, PMT2edge) + r_PMT**2)
    arrive_times = times + (n_water/c)*ts

    write(allowed, events, arrive_times, PETruth)


def write(allowed, events, times, PETruth):
    for PMT_index, photon_index in np.where(allowed):
        PETruth['EventID'].append(events[photon_index])
        PETruth['ChannelID'].append(PMT_index)
        PETruth['PETime'].append(times[photon_index])
        