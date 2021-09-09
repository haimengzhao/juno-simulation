import numpy as np
from tqdm import tqdm
import multiprocessing
import numexpr as ne
from numba import njit
import h5py as h5
import gc
from scipy.spatial import KDTree
from .utils import xyz_from_spher
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
c = 0.3

PETruth = {}
PETruth['EventID'] = []
PETruth['ChannelID'] = []
PETruth['PETime'] = []
PMTs = np.array([[1,2], [1,2], [1,2]])
kdtree = KDTree(PMTs)

def get_PE_Truth(ParticleTruth, PhotonTruth, PMT_list):
    '''
    从ParticleTruth和PhotonTruth, 结合PMT几何信息，给出PETruth
    一次并行计算event_num // 10 个event
    '''
    # 初始化全局变量，我担心跑两遍会出奇怪的问题
    global PETruth, PMTs, kdtree
    PETruth = {}
    PETruth['EventID'] = []
    PETruth['ChannelID'] = []
    PETruth['PETime'] = []

    event_num = ParticleTruth.shape[0]
    PMT_x, PMT_y, PMT_z = xyz_from_spher(
        Ro, PMT_list['theta']/180*np.pi, PMT_list['phi']/180*np.pi
    )
    PMTs = np.stack((PMT_x, PMT_y, PMT_z))
    kdtree = KDTree(np.stack((PMT_x, PMT_y, PMT_z), axis=-1))

    # 生成每个photon的坐标coordinates
    photon_num_per_event = np.roll(PhotonTruth[
        np.unique(PhotonTruth['EventID'], return_index=True)[1] - 1
    ]['PhotonID'], -1) + 1
    repeated_par_tr = np.repeat(ParticleTruth, photon_num_per_event)
    coordinates = np.stack(
        (repeated_par_tr['x']/1000, repeated_par_tr['y']/1000, repeated_par_tr['z']/1000)
    )
    
    # 将问题分成10次循环，以防止内存爆炸
    start_coordinate_index = 0
    step = event_num // 10
    for event_index in tqdm(range(0, step*11, step)):
        photon = np.sum(photon_num_per_event[event_index:event_index+step])
        if photon == 0:
            break
        end_coordinate_index = start_coordinate_index + photon

        chunk_coordinates = coordinates[
            :, start_coordinate_index:end_coordinate_index
        ]

        t = np.random.random(photon) * np.pi
        p = np.random.random(photon) * 2 * np.pi
        vxs = np.sin(t) * np.cos(p)
        vys = np.sin(t) * np.sin(p)
        vzs = np.cos(t)
        try_velocities = np.stack((vxs, vys, vzs))
        events = PhotonTruth[start_coordinate_index:end_coordinate_index]['EventID']
        times = PhotonTruth[start_coordinate_index:end_coordinate_index]['GenTime']
        can_reflect = np.ones(photon)
        transist(chunk_coordinates, try_velocities, times, events, can_reflect)
        start_coordinate_index = end_coordinate_index
 
    # 将dict转成structured array
    names = ['EventID', 'ChannelID', 'PETime']
    formats = ['<i4', '<i4', '<f8']
    dtype = dict(names=names, formats=formats)
    PETruth_structured = np.empty(len(PETruth['EventID']), dtype=dtype)
    PETruth_structured['EventID'] = PETruth['EventID']
    PETruth_structured['ChannelID'] = PETruth['ChannelID']
    PETruth_structured['PETime'] = PETruth['PETime']

    # 按event排序
    order = np.argsort(PETruth_structured['EventID'])
    PETruth_structured = PETruth_structured[order]

    return PETruth_structured


def transist(coordinates, velocities, times, events, can_reflect):
    # 求解折射点
    cv = np.einsum('kn, kn->n', coordinates, velocities)
    ts = -cv + np.sqrt(cv**2 - np.einsum('kn, kn->n', coordinates, coordinates) + Ri**2)   #到达液闪边界的时间
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
    need_reflect = np.logical_not(need_transmit) * can_reflect.astype(bool)

    # 处理需要折射出去的光子
    if need_transmit.any():
        hit_PMT(edge_points[:, need_transmit], new_velocities[:, need_transmit], 
                new_times[need_transmit], events[need_transmit], can_reflect[need_transmit])

    # 处理需要继续反射的光子
    if need_reflect.any():
        transist(edge_points[:, need_reflect], reflected_velocities[:, need_reflect], 
                 new_times[need_reflect], events[need_reflect], np.zeros(need_reflect.sum()))


def find_hit_PMT(coordinates, velocities, fromPMT=False):
    '''
    coordinates: (3, n)
    PMT_coordinates: (3, m)
    接收一族光线，给出其未来所有时间内与给定PMT的最近距离
    注意：光线是有方向的，如果光子将越离越远，那么将返回负数距离
    '''
    # 查找在球面上最邻近的PMT
    cv1 = np.einsum('kn, kn->n', coordinates, velocities)
    ts1 = -cv1 + np.sqrt(cv1**2 - (np.einsum('kn, kn->n', coordinates, coordinates)-Ro**2))    #到达液闪边界的时间
    edge_points1 = coordinates + ts1 * velocities
    outer_points = np.stack((edge_points1[0, :], edge_points1[1, :], edge_points1[2, :]), axis=-1)

    inserted_points = np.empty(1)
    if fromPMT:
        insert_num = 100
        inner_points = np.stack((coordinates[0, :], coordinates[1, :], coordinates[2, :]), axis=-1)
        inserted_points = np.linspace(inner_points, outer_points, insert_num)[1:, :, :]
    else:
        insert_num = 5
        cv2 = np.einsum('kn, kn->n', coordinates, velocities)
        ts2 = -cv2 + np.sqrt(cv2**2 - (np.einsum('kn, kn->n', coordinates, coordinates)-(Ro-r_PMT)**2))    #到达液闪边界的时间
        edge_points2 = coordinates + ts2 * velocities
        inner_points = np.stack((edge_points2[0, :], edge_points2[1, :], edge_points2[2, :]), axis=-1)
        inserted_points = np.linspace(inner_points, outer_points, insert_num)

    search_distances, search_indexs = kdtree.query(inserted_points, workers=-1, distance_upper_bound=r_PMT) # 返回搜索得到的最邻近点距离和最邻近点index
    allowed_distances = search_distances < np.inf
    possible_photon = np.where(np.any(allowed_distances, axis=0))[0]
    first_point_index = np.argmax(allowed_distances[:, possible_photon], axis=0)

    nearest_PMT_index = search_indexs[first_point_index, possible_photon]
    del search_distances, search_indexs, allowed_distances, first_point_index

    return nearest_PMT_index, possible_photon


def hit_PMT(coordinates, velocities, times, events, can_reflect, fromPMT=False):
    # 给出打到的PMT编号和能打中PMT光子的编号
    nearest_PMT_index, possible_photon = find_hit_PMT(coordinates, velocities, fromPMT)
    possible_coordinates = coordinates[:, possible_photon]
    possible_velocities = velocities[:, possible_photon]
    possible_times = times[possible_photon]
    possible_events = events[possible_photon]
    possible_reflect = can_reflect[possible_photon]
    possible_PMT = PMTs[:, nearest_PMT_index]
    print(f'ratio = {possible_photon.shape[0]/times.shape[0]}')

    # 计算到达时间
    PMT2edge = possible_coordinates - possible_PMT
    ts = -np.einsum('kn, kn->n', PMT2edge, possible_velocities) -\
          np.sqrt(np.einsum('kn, kn->n', PMT2edge, possible_velocities)**2 - np.einsum('kn, kn->n', PMT2edge, PMT2edge) + r_PMT**2)
    arrive_times = possible_times + (n_water/c)*ts

    # 计算到达点，以及入射角、出射角
    edge_points = possible_coordinates + ts*possible_velocities
    normal_vectors = (edge_points - possible_PMT) / r_PMT
    norm = np.linalg.norm(normal_vectors, axis=0)
    incidence_vectors = possible_velocities
    vi = np.einsum('kn, kn->n', incidence_vectors, normal_vectors)
    incidence_angles = np.arccos(-np.maximum(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1))
    print(f'incidence_angles = {incidence_angles.mean()}, var = {incidence_angles.var()}')
    emergence_angles = np.arcsin((n_water/n_glass) * np.sin(incidence_angles))
    print(f'emergence_angles = {emergence_angles.mean()}, var = {emergence_angles.var()}')
    
    # 计算反射系数->反射概率
    Rs = ne.evaluate('(sin(emergence_angles - incidence_angles)/sin(emergence_angles + incidence_angles))**2')
    Rp = ne.evaluate('(tan(emergence_angles - incidence_angles)/tan(emergence_angles + incidence_angles))**2')
    R = (Rs+Rp)/2
    print(f'R average = {R.mean()}')

    probs = np.random.random(possible_photon.shape[0])
    need_transmit = probs > R  # 水的折射率小于玻璃，不可能全反射
    need_reflect = np.logical_not(need_transmit) * possible_reflect.astype(bool)

    # 处理折射进入PMT的光子
    if need_transmit.any():
        print(f'need_transmit = {need_transmit.sum()}')
        write(possible_events[need_transmit], nearest_PMT_index[need_transmit], arrive_times[need_transmit], PETruth)

    # 处理需要继续反射的光子
    if need_reflect.any():
        print(f'need_reflect = {need_reflect.sum()}')
        reflect_coordinates = edge_points[:, need_reflect]
        reflect_velocities = incidence_vectors[:, need_reflect] -\
                              2 * np.einsum('kn ,kn->n', incidence_vectors[:, need_reflect], normal_vectors[:, need_reflect]) * normal_vectors[:, need_reflect]
        reflect_times = arrive_times[need_reflect]
        reflect_events = possible_events[need_reflect]

        # 计算反射光线到球心的距离，判断是否会射回液闪内
        rt = -np.einsum('kn, kn->n', reflect_coordinates, reflect_velocities)
        ds = np.linalg.norm(reflect_coordinates + rt*reflect_velocities, axis=0)
        go_into_LS = ds < Ri
        hit_PMT_again = np.logical_not(go_into_LS)

        # 找出会射回液闪球内的
        print(f'go_into_LS = {go_into_LS.sum()}')
        go_inside(reflect_coordinates[:, go_into_LS], reflect_velocities[:, go_into_LS], 
                  reflect_times[go_into_LS], reflect_events[go_into_LS])

        # 找出继续在水中行进的
        print(f'hit_PMT_again = {hit_PMT_again.sum()}')
        hit_PMT(reflect_coordinates[:, hit_PMT_again], reflect_velocities[:, hit_PMT_again], 
                reflect_times[hit_PMT_again], reflect_events[hit_PMT_again], np.zeros(hit_PMT_again.sum()),fromPMT=True)



def go_inside(coordinates, velocities, times, events):
    # 求解折射点
    cv = np.einsum('kn, kn->n', coordinates, velocities)
    ts = -cv - np.sqrt(cv**2 - np.einsum('kn, kn->n', coordinates, coordinates) + Ri**2)   #到达液闪边界的时间
    edge_points = coordinates + ts * velocities
    new_times = times + (n_water/c)*ts

    # 计算入射角，出射角
    normal_vectors = edge_points / Ri
    incidence_vectors = velocities
    vertical_of_incidence = np.maximum(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1)
    incidence_angles = np.arccos(-vertical_of_incidence)

    #计算折射光，反射光矢量与位置
    reflected_velocities = velocities - 2 * vertical_of_incidence * normal_vectors

    delta = ne.evaluate('1 - eta**2 * (1 - vertical_of_incidence**2)')
    new_velocities = ne.evaluate('eta*incidence_vectors - (eta*vertical_of_incidence + sqrt(delta)) * normal_vectors') #取绝对值避免出错

    # 计算折射系数
    emergence_angles = np.arccos(np.minimum(np.einsum('kn, kn->n', new_velocities, -normal_vectors), 1))
    Rs = ne.evaluate('(sin(emergence_angles - incidence_angles)/sin(emergence_angles + incidence_angles))**2')
    Rp = ne.evaluate('(tan(emergence_angles - incidence_angles)/tan(emergence_angles + incidence_angles))**2')
    R = (Rs+Rp)/2

    # 选出需要折射和反射的光子
    probs = np.random.random(times.shape[0])
    need_transmit = probs > R

    if need_transmit.any():
        transist(edge_points[:, need_transmit], new_velocities[:, need_transmit], new_times[need_transmit], events[need_transmit], np.zeros(need_transmit.sum()))




def write(events, PMT_indexs, times, PETruth):
    PETruth['EventID'].extend(list(events))
    PETruth['ChannelID'].extend(list(PMT_indexs))
    PETruth['PETime'].extend(list(times))

