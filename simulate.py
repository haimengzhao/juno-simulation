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
n_glass = 1.5
eta = n_LS / n_water
Ri = 17.71
Ro = 19.5
r_PMT = 0.508
c = 3e8


def transist(coordinates, velocities, intensities, times, need_reflect=False):
    '''
    coordinates: (3,n)
    velocities: (3,n)，其中每个速度矢量都已经归一化
    intensities: (n,)
    '''
    # 求解折射点
    ts = (-np.einsum('kn, kn->n', coordinates, velocities) +\
           np.sqrt(np.einsum('kn, kn, pn, pn->n', coordinates, velocities, coordinates, velocities) -\
          (np.einsum('kn, kn->n', coordinates, coordinates)-Ri**2)))     #到达液闪边界的时间
    edge_points = coordinates + np.einsum('n, kn->kn', ts, velocities)
    
    # 计算增加的时间
    new_times = times + ts/c*n_LS

    # 计算入射角，出射角
    normal_vectors = -edge_points / Ri
    incidence_vectors = velocities
    vertical_of_incidence = np.clip(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1, 1)
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
    emergence_angles = np.arccos(np.clip(np.einsum('kn, kn->n', new_velocities, -normal_vectors), -1, 1))
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    R = (Rs+Rp)/2
    T = 1 - R
    
    # 计算折射光，反射光强度
    new_intensities = np.einsum('n, n, n->n', intensities, T, can_transmit)
    reflected_intensities = np.einsum('n, n, n->n', intensities, R, can_transmit) + all_reflect

    # 输出所有量，按需拿取
    return new_coordinates, new_velocities, new_intensities, new_times, reflected_coordinates, reflected_velocities, reflected_intensities



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


def hit_PMT_intensity(coordinates, velocities, intensities, times, PMT_coordinates):
    # 取出所有能到达PMT的光线
    distances = distance(coordinates, velocities, PMT_coordinates)
    hit_PMT = np.einsum('n, n->n', (lambda x: x>0)(distances), (lambda x: x<r_PMT)(distances))
    allowed_index = np.where(hit_PMT)[0]
    allowed_coordinates = coordinates[:, allowed_index]
    allowed_velocities = velocities[:, allowed_index]
    allowed_intensities = intensities[allowed_index]
    allowed_times = times[allowed_index]
    allowed_PMT_coordinates = PMT_coordinates[:, :allowed_index.shape[0]]
    # Bonus: 考虑PMT表面的反射
    PMT2edge = allowed_coordinates - allowed_PMT_coordinates

    # # 用正弦定理计算入射角
    # phis = np.arccos(np.sqrt(np.einsum('kn, kn->n', allowed_velocities, edge2PMT) / norm2PMT))
    # incidence_angles = np.arcsin(np.einsum('n, n->n', norm2PMT, np.sin(phis))/r_PMT)
    # emergence_angles = np.arcsin(n_water/n_glass * np.sin(incidence_angles))
    # Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    # Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    # R = (Rs+Rp)/2
    # T = 1 - R

    # 计算到达时间
    ts = -np.einsum('kn, kn->n', PMT2edge, allowed_velocities) +\
          np.sqrt(np.einsum('kn, kn->n', PMT2edge, allowed_velocities)**2 - np.einsum('kn, kn->n', PMT2edge, PMT2edge) + r_PMT**2)
    all_times = allowed_times + ts/c*n_water
    edge_points = allowed_coordinates + np.einsum('n, kn->kn', ts, allowed_velocities)

    # 计算入射角，出射角
    normal_vectors = (edge_points - allowed_PMT_coordinates) / r_PMT
    incidence_vectors = allowed_velocities
    vertical_of_incidence = np.clip(np.einsum('kn, kn->n', incidence_vectors, normal_vectors), -1, 1)
    incidence_angles = np.arccos(-vertical_of_incidence)

    # Bonus: 计算进入PMT的折射系数
    emergence_angles = np.arccos(np.clip(np.einsum('kn, kn->n', allowed_velocities, -normal_vectors), -1, 1))
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    R = (Rs+Rp)/2
    T = 1 - R

    all_intensity = np.einsum('n, n->', allowed_intensities, T)

    return all_intensity, all_times


try_num = 20000
try_phis = np.random.rand(try_num) * 2 * np.pi
try_thetas = np.arccos(np.random.rand(try_num)*2 - 1)

vxs = np.einsum('n, n->n', np.sin(try_thetas), np.cos(try_phis))
vys = np.einsum('n, n->n', np.sin(try_thetas), np.sin(try_phis))
vzs = np.cos(try_thetas)
try_velocities = np.stack((vxs, vys, vzs))
try_intensities = np.ones(try_num)

def get_PE_probability(x, y, z, PMT_phi, PMT_theta, naive=False):
    # Step0： 将PMT转到(pi, pi/2)处
    if PMT_phi != np.pi or PMT_theta != np.pi/2:
        Rz = np.array([[-np.cos(PMT_phi), -np.sin(PMT_phi), 0],
                       [ np.sin(PMT_phi), -np.cos(PMT_phi), 0],
                       [               0,                0, 1]])
        Ry = np.array([[np.sin(PMT_theta), 0, -np.cos(PMT_theta)],
                       [                0, 1,                  0],
                       [np.cos(PMT_theta), 0,  np.sin(PMT_theta)]])
        nx, ny, nz = Ry @ Rz @ np.array((x, y, z))
        return get_PE_probability(nx, ny, nz, np.pi, np.pi/2)
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
    vertex2PMT = np.sqrt((x-PMT_x)**2+(y-PMT_y)**2+(z-PMT_z)**2)
    d_min = 0.510
    d_max = 0.8 + vertex2PMT*0.04
    allowed_lights = np.einsum('n, n->n', (lambda x: x>d_min)(try_distances), (lambda x: x<d_max)(try_distances))
    valid_index = np.where(allowed_lights)[0]
    allow_num = valid_index.shape[0] #如果大于400， 则说明顶点与PMT非常接近
    #print(f'allowed = {allow_num}')
    allowed_phis = try_phis[valid_index]
    phi_start = allowed_phis.min()
    phi_end = allowed_phis.max()
    allowed_thetas = try_thetas[valid_index]
    theta_start = allowed_thetas.min()
    theta_end = allowed_thetas.max()
    
    
    # phi_start, phi_end = 0, 2*np.pi
    # theta_start = theta_min if theta_min>0.1 else 0.0001
    # theta_end = theta_max if (2*np.pi - theta_max)>0.1 else np.pi-0.0001
    # if theta_start>0.1 and allow_num>100 and z>0:       # 修正特别贴近的情况
    #     theta_start = 0.0001
    # if (2*np.pi - theta_max)>0.1 and allow_num>100 and z<0:
    #     theta_end = np.pi-0.0001

    Omega = (np.cos(theta_start) - np.cos(theta_end)) * (phi_end - phi_start)
    print(f'phi in {[phi_start, phi_end]}')
    print(f'theta in {[theta_start, theta_end]}')
    # print(f'Omega = {Omega}')
    
    # Step3: 在小区域中选择光线
    dense_phi_num = 500
    dense_theta_num = 500
    dense_phis = np.linspace(phi_start, phi_end, dense_phi_num)
    dense_thetas = np.arccos(np.linspace(np.cos(theta_start), np.cos(theta_end), dense_theta_num))

    dense_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, x, y, z)
    dense_velocities = gen_velocities(dense_phis, dense_thetas)
    dense_intensities = np.ones(dense_phi_num*dense_theta_num)
    dense_times = np.zeros(dense_phi_num*dense_theta_num)

    # Step4: 判断哪些光线能够到达PMT
    dense_new_coordinates, dense_new_velocities, dense_new_intensities, dense_new_times= transist(dense_coordinates, dense_velocities, dense_intensities, dense_times)[:4]
    dense_PMT_coordinates = gen_coordinates(dense_phi_num*dense_theta_num, PMT_x, PMT_y, PMT_z)
    all_intensity, all_times = hit_PMT_intensity(dense_new_coordinates, dense_new_velocities, dense_new_intensities, dense_new_times, dense_PMT_coordinates)
    ratio = all_intensity / (dense_phi_num*dense_theta_num)
    # print(f'ratio = {ratio}')
    prob = ratio * Omega / (4*np.pi)
    print(f'prob = {prob}')
    print(f'transist time = {all_times.mean()}')
    return prob, all_times

x = np.random.rand(4000) * 10
y = np.random.rand(4000) * 10
z = np.random.rand(4000) * 10
ti = time()
# pool = multiprocessing.Pool(processes=7)

# for step in range(4000):
#     s = pool.apply_async(get_PE_probability, (x[step], y[step], z[step], 0, 0))

# pool.close()
# pool.join()
get_PE_probability(3,6,10,0,0)
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
