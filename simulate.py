import argparse
import numpy as np
from time import time
import multiprocessing

# # 处理命令行
# parser = argparse.ArgumentParser()
# parser.add_argument("-n", dest="n", type=int, help="Number of events")
# parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
# parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
# args = parser.parse_args()

import h5py as h5

# TODO: 模拟顶点
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
    raise NotImplementedError

# TODO: 光学部分
n_water = 1.33
n_LS = 1.48
eta = n_LS / n_water
Ri = 17.71
Ro = 19.5
r_PMT = 0.508


def transist(coordinates, velocities, intensities, need_reflect=False):
    '''
    coordinates: (3,n)
    velocities: (3,n)，其中每个速度矢量都已经归一化
    intensities: (n,)
    '''
    # 求解折射点
    ts = (-np.einsum('kn, kn->n', coordinates, velocities) +\
          np.sqrt(np.einsum('kn, kn, pn, pn->n', coordinates, velocities, coordinates, velocities) -\
          np.einsum('kn, kn->n', velocities, velocities) * (np.einsum('kn, kn->n', coordinates, coordinates)-Ri**2))) /\
          np.einsum('kn, kn->n', velocities, velocities)  #到达液闪边界的时间
    edge_points = coordinates + np.einsum('n, kn->kn', ts, velocities)

    # 计算入射角，出射角
    normal_vectors = -edge_points / Ri
    incidence_vectors = velocities
    vertical_of_incidence = np.einsum('kn, kn->n', incidence_vectors, normal_vectors)
    incidence_angles = np.arccos(-vertical_of_incidence)
    
    
    # 判断全反射
    max_incidence_angle = np.arcsin(n_water/n_LS)
    can_transmit = (lambda x: x<max_incidence_angle)(incidence_angles)
    all_reflect = 1 - can_transmit

    #计算折射光，反射光矢量与位置
    reflected_velocities = velocities - 2 * np.einsum('n, kn->kn', vertical_of_incidence, normal_vectors)
    reflected_coordinates = edge_points

    delta = 1 - eta**2 * (1 - vertical_of_incidence**2)
    new_velocities = eta*incidence_vectors -\
                     np.einsum('n, kn->kn', eta*vertical_of_incidence + np.sqrt(np.abs(delta)), normal_vectors) #取绝对值避免出错
    new_velocities = np.einsum('n, kn->kn', can_transmit, new_velocities)
    new_coordinates = edge_points

    # 计算折射系数
    emergence_angles = np.arccos(np.einsum('kn, kn->n', new_velocities, -normal_vectors))
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    R = (Rs+Rp)/2
    T = 1 - R
    
    # 计算折射光，反射光强度
    new_intensities = np.einsum('n, n, n->n', intensities, T, can_transmit)
    reflected_intensities = np.einsum('n, n, n->n', intensities, R, can_transmit) + all_reflect

    # 输出所有量，按需拿取
    return new_coordinates, new_velocities, new_intensities, reflected_coordinates, reflected_velocities, reflected_intensities



def distance(coordinates, velocities, PMT_coordinates):
    new_ts = np.einsum('kn, kn->n', PMT_coordinates - coordinates, velocities)
    nearest_points = coordinates + np.einsum('n, kn->kn', new_ts, velocities)
    distances = np.sqrt(np.einsum('kn, kn->n', nearest_points - PMT_coordinates, nearest_points - PMT_coordinates)) *\
                (lambda x: x>0)(new_ts)
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
    vxs = np.einsum('n, n->n', np.sin(theta_d), np.cos(phi_d))
    vys = np.einsum('n, n->n', np.sin(theta_d), np.sin(phi_d))
    vzs = np.cos(theta_d)
    velocities = np.stack((vxs, vys, vzs))
    return velocities


try_num = 20000
try_phis = np.random.rand(try_num) * 2 * np.pi
try_thetas = np.arccos(np.random.rand(try_num)*2 - 1)

vxs = np.einsum('n, n->n', np.sin(try_thetas), np.cos(try_phis))
vys = np.einsum('n, n->n', np.sin(try_thetas), np.sin(try_phis))
vzs = np.cos(try_thetas)
try_velocities = np.stack((vxs, vys, vzs))
try_intensities = np.ones(try_num)

def get_PE_probability(x, y, z, PMT_phi, PMT_theta):
    # 读取PMT坐标信息
    PMT_x = Ro * np.sin(PMT_theta) * np.cos(PMT_phi)
    PMT_y = Ro * np.sin(PMT_theta) * np.sin(PMT_phi)
    PMT_z = Ro * np.cos(PMT_theta)
    
    # Step1: 均匀发出试探光线
    try_coordinates = gen_coordinates(try_num, x, y, z)

    # Step2: 寻找距离PMT中心一定距离的折射光
    try_new_coordinates, try_new_velocities, ti, try_reflected_coordinates, try_reflected_velocities = transist(try_coordinates, try_velocities, try_intensities)[:5]
    try_PMT_coordinates = gen_coordinates(try_num, PMT_x, PMT_y, PMT_z)
    try_distances = distance(try_new_coordinates, try_new_velocities, try_PMT_coordinates)

    d_min = 0.510
    d_max = 1
    allowed_lights = np.einsum('n, n->n', (lambda x: x>d_min)(try_distances), (lambda x: x<d_max)(try_distances))
    valid_index = np.where(allowed_lights)[0]
    print(allowed_lights.sum())
    allowed_thetas = try_thetas[valid_index]
    theta_min = allowed_thetas.min()
    theta_max = allowed_thetas.max()
    
    phi_start, phi_end = 0, 2*np.pi
    theta_start = theta_min if theta_min>0.1 else 0.0001
    theta_end = theta_max if (2*np.pi - theta_max)>0.1 else 2*np.pi-0.0001
    Omega = (np.cos(theta_start) - np.cos(theta_end)) * 2 * np.pi
    #print(theta_end- theta_start)
    #print(f'Omega = {Omega}')

    # Step3: 在小区域中选择光线
    dense_phi_num = 2000
    dense_theta_num = 100
    dense_phis = np.linspace(phi_start, phi_end, dense_phi_num)
    dense_thetas = np.arccos(np.linspace(np.cos(theta_start), np.cos(theta_end), dense_theta_num))

    dense_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, x, y, z)
    dense_velocities = gen_velocities(dense_phis, dense_thetas)
    dense_intensities = np.ones(dense_phi_num*dense_theta_num)

    # Step4: 判断哪些光线能够到达PMT
    dense_new_coordinates, dense_new_velocities, dense_new_intensities = transist(dense_coordinates, dense_velocities, dense_intensities)[:3]
    dense_PMT_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, PMT_x, PMT_y, PMT_z)
    dense_distances = distance(dense_new_coordinates, dense_new_velocities, dense_PMT_coordinates)
    hit_PMT_num = np.einsum('n, n->n', (lambda x: x>0)(dense_distances), (lambda x: x<r_PMT)(dense_distances))
    all_intensity = np.einsum('n, n->', dense_new_intensities, hit_PMT_num)
    ratio = all_intensity / (dense_phi_num*dense_theta_num)
    #print(f'ratio = {ratio}')

    prob = ratio * Omega / (4*np.pi)
    #print(f'prob = {prob}')
    return prob

x = np.random.rand(4000) * 10
y = np.random.rand(4000) * 10
z = np.random.rand(4000) * 10
pool = multiprocessing.Pool(processes=8)
ti = time()
for step in range(4000):
    pool.apply_async(get_PE_probability, (x[step], y[step], z[step], 0, 0))
#get_PE_probability(0,0,0.1,0,0)
pool.close()
pool.join()
to = time()
print(f'time = {to-ti}')
# # 读入几何文件
# with h5.File(args.geo, "r") as geo:
#     # 只要求模拟17612个PMT
#     PMT_list = geo['Geometry'][:17612]

# # 输出
# with h5.File(args.opt, "w") as opt:
#     # 生成顶点
#     ParticleTruth, PhotonTruth = generate_events(args.n)
    
#     opt['ParticleTruth'] = ParticleTruth

    
#     print("TODO: Write opt file")
