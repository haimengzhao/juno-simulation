import numpy as np
from time import time
import multiprocessing
from timeit import Timer
import numexpr as ne

ne.set_num_threads(2)

# TODO: 光学部分
n_water = 1.33
n_LS = 1.48
n_glass = 1.5
eta = n_LS / n_water
Ri = 17.71
Ro = 19.5
r_PMT = 0.508
c = 3e8


def transist_once(coordinates, velocities, intensities, times):
    '''
    coordinates: (3,n)
    velocities: (3,n)，其中每个速度矢量都已经归一化
    intensities: (n,)
    '''
    # 求解折射点
    ts = (-np.einsum('kn, kn->n', coordinates, velocities) +\
           np.sqrt(np.einsum('kn, kn->n', coordinates, velocities)**2 -\
          (np.einsum('kn, kn->n', coordinates, coordinates)-Ri**2)))     #到达液闪边界的时间
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
    reflected_coordinates = edge_points

    delta = ne.evaluate('1 - eta**2 * (1 - vertical_of_incidence**2)')
    new_velocities = ne.evaluate('(eta*incidence_vectors - (eta*vertical_of_incidence + sqrt(abs(delta))) * normal_vectors) * can_transmit') #取绝对值避免出错
    new_coordinates = edge_points

    # 计算折射系数
    emergence_angles = np.arccos(np.minimum(np.einsum('kn, kn->n', new_velocities, -normal_vectors), 1))
    Rs = ne.evaluate('(sin(emergence_angles - incidence_angles)/sin(emergence_angles + incidence_angles))**2')
    Rp = ne.evaluate('(tan(emergence_angles - incidence_angles)/tan(emergence_angles + incidence_angles))**2')
    R = (Rs+Rp)/2
    T = 1 - R
    
    # 计算折射光，反射光强度
    new_intensities = np.einsum('n, n, n->n', intensities, T, can_transmit)
    reflected_intensities = np.einsum('n, n, n->n', intensities, R, can_transmit) + all_reflect

    # 输出所有量，按需拿取
    return new_coordinates, new_velocities, new_intensities, new_times, reflected_coordinates, reflected_velocities, reflected_intensities


def transist_twice(coordinates, velocities, intensities, times):
    nt, nc, nv, ni = transist_once(coordinates, velocities, intensities, times)[3:]
    return transist_once(nc, nv, ni, nt)
    

def distance(coordinates, velocities, PMT_coordinates):
    new_ts = np.einsum('kn, kn->n', PMT_coordinates - coordinates, velocities)
    nearest_points = coordinates + new_ts * velocities
    distances = np.linalg.norm(nearest_points - PMT_coordinates, axis=0) * (new_ts>0)
    return distances


def gen_coordinates(len, x, y, z):
    coordinate_x = np.full(len, x)
    coordinate_y = np.full(len, y)
    coordinate_z = np.full(len, z)
    coordinates = np.stack((coordinate_x, coordinate_y, coordinate_z))
    return coordinates

def gen_velocities(phis, thetas):
    vxs = (np.sin(thetas) * np.cos(phis.reshape(-1, 1))).reshape(-1)
    vys = (np.sin(thetas) * np.sin(phis.reshape(-1, 1))).reshape(-1)
    vzs = np.tile(np.cos(thetas), phis.shape[0])
    velocities = np.stack((vxs, vys, vzs))
    return velocities


def hit_PMT(coordinates, velocities, intensities, times, PMT_coordinates):
    # 取出所有能到达PMT的光线
    distances = distance(coordinates, velocities, PMT_coordinates)
    hit_PMT = (distances>0) * (distances<r_PMT)
    allowed_coordinates = coordinates[:, hit_PMT]
    allowed_velocities = velocities[:, hit_PMT]
    allowed_intensities = intensities[hit_PMT]
    allowed_times = times[hit_PMT]
    allowed_PMT_coordinates = PMT_coordinates[:, :allowed_times.shape[0]]
   
    # 计算到达时间
    PMT2edge = allowed_coordinates - allowed_PMT_coordinates
    ts = -np.einsum('kn, kn->n', PMT2edge, allowed_velocities) +\
          np.sqrt(np.einsum('kn, kn->n', PMT2edge, allowed_velocities)**2 - np.einsum('kn, kn->n', PMT2edge, PMT2edge) + r_PMT**2)
    all_times = allowed_times + (n_water/c)*ts
    edge_points = allowed_coordinates + ts * allowed_velocities

    # 计算入射角，出射角
    normal_vectors = (edge_points - allowed_PMT_coordinates) / r_PMT
    incidence_vectors = allowed_velocities
    incidence_angles = np.arccos(-np.maximum(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1))

    # Bonus: 计算进入PMT的折射系数
    emergence_angles = np.arccos(np.minimum(np.einsum('kn, kn->n', allowed_velocities, -normal_vectors), 1))
    Rs = ne.evaluate('(sin(emergence_angles - incidence_angles)/sin(emergence_angles + incidence_angles))**2')
    Rp = ne.evaluate('(tan(emergence_angles - incidence_angles)/tan(emergence_angles + incidence_angles))**2')
    R = (Rs+Rp)/2
    T = 1 - R

    all_intensity = np.einsum('n, n->', allowed_intensities, T)

    return all_intensity, all_times

def rotate(x, y, z, PMT_phi, PMT_theta, reflect_num):
    if reflect_num == 0:
        if PMT_phi != np.pi or PMT_theta != np.pi/2:
            Rz = np.array([[-np.cos(PMT_phi), -np.sin(PMT_phi), 0],
                           [ np.sin(PMT_phi), -np.cos(PMT_phi), 0],
                           [               0,                0, 1]])
            Ry = np.array([[np.sin(PMT_theta), 0, -np.cos(PMT_theta)],
                           [                0, 1,                  0],
                           [np.cos(PMT_theta), 0,  np.sin(PMT_theta)]])
            nx, ny, nz = Ry @ Rz @ np.array((x, y, z))
            return nx, ny, nz, np.pi, np.pi/2
        else:
            return x, y, z, np.pi, np.pi/2
    elif reflect_num == 1:
        if PMT_phi != 0 or PMT_theta != np.pi/2:
            Rz = np.array([[ np.cos(PMT_phi), np.sin(PMT_phi), 0],
                           [-np.sin(PMT_phi), np.cos(PMT_phi), 0],
                           [               0,               0, 1]])
            Ry = np.array([[ np.sin(PMT_theta), 0, np.cos(PMT_theta)],
                           [                 0, 1,                 0],
                           [-np.cos(PMT_theta), 0, np.sin(PMT_theta)]])
            nx, ny, nz = Ry @ Rz @ np.array((x, y, z))
            return nx, ny, nz, 0, np.pi/2
        else:
            return x, y, z, 0, np.pi/2

try_num = 100
try_phi = np.linspace(0, 2*np.pi, try_num, endpoint=False)
try_theta = np.arccos(np.linspace(1, -1, try_num))
try_phis, try_thetas = np.meshgrid(try_phi, try_theta)
try_phis = try_phis.flatten() + np.random.random(try_num**2) / try_num**2
try_thetas= try_thetas.flatten() + np.random.random(try_num**2) / try_num**2

vxs = np.sin(try_thetas) * np.cos(try_phis)
vys = np.sin(try_thetas) * np.sin(try_phis)
vzs = np.cos(try_thetas)
try_velocities = np.stack((vxs, vys, vzs))

# try_velocities = gen_velocities(try_phi, try_theta)
try_intensities = np.ones(try_num**2)

def get_prob_time(x, y, z, PMT_phi, PMT_theta, reflect_num, acc):
    if reflect_num == 0:
        transist = transist_once
    elif reflect_num == 1:
        transist = transist_twice
    # Step0： 将PMT转到(pi, pi/2)处
    x, y, z, PMT_phi, PMT_theta = rotate(x, y, z, PMT_phi, PMT_theta, reflect_num)
    # 读取PMT坐标信息
    PMT_x = Ro * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y = Ro * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z = Ro * np.cos(PMT_theta)

    # Step1: 均匀发出试探光线
    try_coordinates = gen_coordinates(try_num**2, x, y, z)
    try_times = np.zeros(try_num**2)

    # Step2: 寻找距离PMT中心一定距离的折射光
    try_new_coordinates, try_new_velocities = transist(try_coordinates, try_velocities, try_intensities, try_times)[:2]
    try_PMT_coordinates = gen_coordinates(try_num**2, PMT_x, PMT_y, PMT_z)
    try_distances = distance(try_new_coordinates, try_new_velocities, try_PMT_coordinates)

    d_min = 0.510
    # 自动调节d_max，使得粗调得到一个恰当的范围（20根粗射光线）
    allow_num = 0
    least_allow_num = 12
    for d_max in np.linspace(0.6, 5, 100):
        allowed_lights = (try_distances>d_min) * (try_distances<d_max)
        allow_num = np.sum(allowed_lights)
        if allow_num > least_allow_num:
            break
    if allow_num <= least_allow_num:
        return 0, np.zeros(1)
    
    # print(f'dmax = {d_max}')
    # print(f'allowed = {allow_num}')
    allowed_phis = try_phis[allowed_lights]
    phi_start = allowed_phis.min()
    phi_end = allowed_phis.max()
    allowed_thetas = try_thetas[allowed_lights]
    theta_start = allowed_thetas.min()
    theta_end = allowed_thetas.max()

    Omega = (np.cos(theta_start) - np.cos(theta_end)) * (phi_end - phi_start)
    # print(f'phi in {[phi_start, phi_end]}')
    # print(f'theta in {[theta_start, theta_end]}')
    # print(f'Omega = {Omega}')
    
    # Step3: 在小区域中选择光线
    dense_phi_num = acc
    dense_theta_num = acc
    dense_phis = np.linspace(phi_start, phi_end, dense_phi_num)
    dense_thetas = np.arccos(np.linspace(np.cos(theta_start), np.cos(theta_end), dense_theta_num))

    dense_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, x, y, z)
    dense_velocities = gen_velocities(dense_phis, dense_thetas)
    dense_intensities = np.ones(dense_phi_num*dense_theta_num)
    dense_times = np.zeros(dense_phi_num*dense_theta_num)

    # Step4: 判断哪些光线能够到达PMT
    dense_new_coordinates, dense_new_velocities, dense_new_intensities, dense_new_times= transist(dense_coordinates, dense_velocities, dense_intensities, dense_times)[:4]
    dense_PMT_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, PMT_x, PMT_y, PMT_z)
    all_intensity, all_times = hit_PMT(dense_new_coordinates, dense_new_velocities, dense_new_intensities, dense_new_times, dense_PMT_coordinates)
    ratio = all_intensity / (dense_phi_num*dense_theta_num)
    # print(f'light num = {all_times.shape[0]}')
    # print(f'ratio = {ratio}')
    prob = ratio * Omega / (4*np.pi)
    # print(f'prob = {prob}')
    # print(f'transist time = {all_times.mean()}')
    return prob, all_times


def get_PE_probability(x, y, z, PMT_phi, PMT_theta, naive=False):
    PMT_x = Ro * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y = Ro * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z = Ro * np.cos(PMT_theta)
    d = np.sqrt((x-PMT_x)**2 + (y-PMT_y)**2 + (z-PMT_z)**2)
    if naive:
        return r_PMT**2/(4*d**2)  # 平方反比模式
    else:
        prob1 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 0, 150)[0]
        prob2 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 1, 100)[0]
        # print(prob1)
        # print(prob2)
        return prob1 + prob2

def get_random_PE_time(x, y, z, PMT_phi, PMT_theta):
    prob1, times1 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 0, 100)
    prob2, times2 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 1, 50)
    # print(prob1/prob2)
    p = np.random.rand()
    if p < prob1/(prob1+prob2):     # 即一次折射无反射
        return np.random.choice(times1)
    else:
        return np.random.choice(times2)




# ti = time()
# pool = multiprocessing.Pool(processes=7)

# for step in range(4000):
#     s = pool.apply_async(get_PE_probability, (x[step], y[step], z[step], 0, 0))

# pool.close()
# pool.join()
# print(get_random_PE_time(3,6,-10,0,0))
# to = time()
# print(f'time = {to-ti}')
x = np.random.random(200) * 10
y = np.random.random(200) * 10
z = np.random.random(200) * 10

if __name__ == '__main__':
    print(Timer('get_PE_probability(3,6,10,0,0)', setup='from __main__ import get_PE_probability').timeit(400))
    # for i in range(200):
    #    get_PE_probability(x[i], y[i], z[i],0,0)
    # print(get_PE_probability(3, 6, 10,0,0))
    # get_PE_probability(np.random.rand()*10, np.random.rand()*10, np.random.rand()*10,0,0)
    # for i in range(4000):
    #     try_phis = np.random.rand(try_num) * 2 * np.pi
    #     try_thetas = np.arccos(np.random.rand(try_num)*2 - 1)

    #     vxs = np.sin(try_thetas) * np.cos(try_phis)
    #     vys = np.sin(try_thetas) * np.sin(try_phis)
    #     vzs = np.cos(try_thetas)
    #     try_velocities = np.stack((vxs, vys, vzs))
    #     try_intensities = np.ones(try_num)
    #     res = get_PE_probability(3, 6, 10,0,0)
    #     print(res)
    #     if np.abs(res*1000000-494)> 10:
    #         print("error", res)
    #         break