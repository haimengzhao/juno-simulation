import numpy as np
from time import time
import multiprocessing
from numba import jit
import numba
from tqdm.std import tqdm


# TODO: 光学部分
n_water = 1.33
n_LS = 1.48
n_glass = 1.5
eta = n_LS / n_water
Ri = 17.71
Ro = 19.5
r_PMT = 0.508
c = 3e8

@jit(nopython=True, parallel=True)
def transist_once(coordinates, velocities, intensities, times):
    '''
    coordinates: (3,n)
    velocities: (3,n)，其中每个速度矢量都已经归一化
    intensities: (n,)
    '''
    # 求解折射点
    ts = -np.sum(coordinates*velocities, axis=0) +\
          np.sqrt(np.sum(coordinates*velocities, axis=0)**2 -\
                 (np.sum(coordinates**2, axis=0)-Ri**2))     #到达液闪边界的时间
    edge_points = coordinates + ts*velocities
    
    # 计算增加的时间
    new_times = times + ts/c*n_LS

    # 计算入射角，出射角
    normal_vectors = -edge_points / Ri
    incidence_vectors = velocities
    vertical_of_incidence = np.minimum(np.maximum(np.sum(incidence_vectors * normal_vectors, axis=0), -1), 1)
    incidence_angles = np.arccos(-vertical_of_incidence)
    
    
    # 判断全反射
    max_incidence_angle = np.arcsin(n_water/n_LS)
    can_transmit = (lambda x: x<max_incidence_angle)(incidence_angles)
    all_reflect = 1 - can_transmit

    #计算折射光，反射光矢量与位置
    reflected_velocities = velocities - 2 * vertical_of_incidence * normal_vectors
    reflected_coordinates = edge_points

    delta = 1 - eta**2 * (1 - vertical_of_incidence**2)
    new_velocities = eta*incidence_vectors -\
                     ((eta*vertical_of_incidence + np.sqrt(np.abs(delta))) * normal_vectors) #取绝对值避免出错
    new_velocities = can_transmit * new_velocities
    new_coordinates = edge_points

    # 计算折射系数
    emergence_angles = np.arccos(np.minimum(np.maximum(-np.sum(new_velocities*normal_vectors, axis=0), -1), 1))
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    R = (Rs+Rp)/2
    T = 1 - R
    
    # 计算折射光，反射光强度
    new_intensities = intensities * T * can_transmit
    reflected_intensities = intensities * R * can_transmit + all_reflect

    # 输出所有量，按需拿取
    return new_coordinates, new_velocities, new_intensities, new_times, reflected_coordinates, reflected_velocities, reflected_intensities


def transist_twice(coordinates, velocities, intensities, times):
    nt, nc, nv, ni = transist_once(coordinates, velocities, intensities, times)[3:]
    return transist_once(nc, nv, ni, nt)
    



def distance(coordinates, velocities, PMT_coordinates):
    new_ts = np.sum((PMT_coordinates - coordinates)*velocities, axis=0)
    nearest_points = coordinates + new_ts * velocities
    distances = np.sqrt(np.sum((nearest_points - PMT_coordinates)**2, axis=0)) * (new_ts>0)
    return distances


def gen_coordinates(len, x, y, z):
    coordinate_x = np.full(len, x)
    coordinate_y = np.full(len, y)
    coordinate_z = np.full(len, z)
    coordinates = np.stack((coordinate_x, coordinate_y, coordinate_z))
    return coordinates

def gen_velocities(phis, thetas):
    phi, theta = np.meshgrid(phis, thetas)
    phi_d = phi.ravel()
    theta_d = theta.ravel()
    vxs = np.sin(theta_d) * np.cos(phi_d)
    vys = np.sin(theta_d) * np.sin(phi_d)
    vzs = np.cos(theta_d)
    velocities = np.stack((vxs, vys, vzs))
    return velocities


def hit_PMT(coordinates, velocities, intensities, times, PMT_coordinates):
    # 取出所有能到达PMT的光线
    distances = distance(coordinates, velocities, PMT_coordinates)
    hit_PMT = (distances>0) * (distances<r_PMT)
    allowed_index = np.where(hit_PMT)[0]
    allowed_coordinates = coordinates[:, allowed_index]
    allowed_velocities = velocities[:, allowed_index]
    allowed_intensities = intensities[allowed_index]
    allowed_times = times[allowed_index]
    allowed_PMT_coordinates = PMT_coordinates[:, :allowed_index.shape[0]]
   
    # 计算到达时间
    PMT2edge = allowed_coordinates - allowed_PMT_coordinates
    ts = -np.sum(PMT2edge*allowed_velocities, axis=0) +\
          np.sqrt(np.sum(PMT2edge*allowed_velocities, axis=0)**2 - np.sum(PMT2edge**2, axis=0) + r_PMT**2)
    all_times = allowed_times + ts/c*n_water
    edge_points = allowed_coordinates + ts * allowed_velocities

    # 计算入射角，出射角
    normal_vectors = (edge_points - allowed_PMT_coordinates) / r_PMT
    incidence_vectors = allowed_velocities
    vertical_of_incidence = np.minimum(np.maximum(np.sum(incidence_vectors*normal_vectors, axis=0), -1), 1)
    incidence_angles = np.arccos(-vertical_of_incidence)

    # Bonus: 计算进入PMT的折射系数
    emergence_angles = np.arccos(np.minimum(np.maximum(-np.sum(allowed_velocities*normal_vectors, axis=0), -1), 1))
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    R = (Rs+Rp)/2
    T = 1 - R

    all_intensity = np.sum(allowed_intensities*T)

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

try_num = 20000
try_phis = np.random.rand(try_num) * 2 * np.pi
try_thetas = np.arccos(np.random.rand(try_num)*2 - 1)

vxs = np.sin(try_thetas) * np.cos(try_phis)
vys = np.sin(try_thetas) * np.sin(try_phis)
vzs = np.cos(try_thetas)
try_velocities = np.stack((vxs, vys, vzs))
try_intensities = np.ones(try_num)

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
    try_coordinates = gen_coordinates(try_num, x, y, z)
    try_times = np.zeros(try_num)

    # Step2: 寻找距离PMT中心一定距离的折射光
    try_new_coordinates, try_new_velocities = transist(try_coordinates, try_velocities, try_intensities, try_times)[:2]
    try_PMT_coordinates = gen_coordinates(try_num, PMT_x, PMT_y, PMT_z)
    try_distances = distance(try_new_coordinates, try_new_velocities, try_PMT_coordinates)

    d_min = 0.510
    for d_max in np.linspace(0.6, 5, 100):
        global valid_index, allow_num
        allowed_lights = (try_distances>d_min) * (try_distances<d_max)
        valid_index = np.where(allowed_lights)[0]
        allow_num = valid_index.shape[0]
        if allow_num > 20:
            break
    if valid_index.shape[0] <= 20:
        return 0, np.zeros(1)
    # print(f'allowed = {allow_num}')
    allowed_phis = try_phis[valid_index]
    phi_start = allowed_phis.min()
    phi_end = allowed_phis.max()
    allowed_thetas = try_thetas[valid_index]
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
        prob1 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 0, 500)[0]
        prob2 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 1, 200)[0]
        # print(prob1)
        # print(prob2)
        return prob1 + prob2

def get_random_PE_time(x, y, z, PMT_phi, PMT_theta):
    prob1, times1 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 0, 500)
    prob2, times2 = get_prob_time(x, y, z, PMT_phi, PMT_theta, 1, 200)
    # print(prob1/prob2)
    p = np.random.rand()
    if p < prob1/(prob1+prob2):     # 即一次折射无反射
        return np.random.choice(times1)
    else:
        return np.random.choice(times2)




ti = time()
# pool = multiprocessing.Pool(processes=7)

# for step in range(4000):
#     s = pool.apply_async(get_PE_probability, (x[step], y[step], z[step], 0, 0))
num = 4000
# pool.close()
# pool.join()
pbar = tqdm(total=num)
def task():
    for i in numba.prange(num):
        get_random_PE_time(3,6,-10,0,0)
        pbar.update()
task()
to = time()
print(f'time = {to-ti}')
