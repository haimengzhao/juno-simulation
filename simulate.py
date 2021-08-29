import argparse
import numpy as np
from scipy.optimize import minimize

# 处理命令行
parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="n", type=int, help="Number of events")
parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
args = parser.parse_args()

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

def distance(x, y):
    return np.sqrt(np.sum(np.square(x-y), axis=-1))

def gen_optical_path_func(vertex_coordinate, PMT_coordinate):
    return lambda y: n_LS*distance(y, vertex_coordinate) +\
                     n_water*distance(y, PMT_coordinate)
            

def get_PE_probability(vertices, PMT_phi, PMT_theta):
    '''
    描述：并行计算(x, y, z)处产生的光子到达(phi, theta)处PMT的概率
    输入：vertices形状为(n, 3)，vertices[i]为第i个顶点的坐标(x, y, z)
          x, y, z: 顶点坐标/m
          phi, theta: PMT坐标
    输出：np.array，每个元素为float64类型的概率
    算法描述：
    1. 计算从(x, y, z)到PMT可能的两条光路：折射/反射+折射
       可能的思路：费马原理？
    2. 计算这两条光路的总长度
    3. 用菲涅尔公式计算假设光子正好沿着这个方向，真的会这样走的概率
    4. 返回3中概率*(求和 PMT的有效截面/光路总长度^2)/4pi立体角
    '''
    S = 4*np.pi * 0.508**2
    PMT_coordinate = 19.5 * np.array([np.sin(PMT_theta)*np.cos(PMT_phi), np.sin(PMT_theta)*np.sin(PMT_phi), np.cos(PMT_theta)])
    con = {'type':'eq', 'fun':lambda x: np.square(x).sum()-17.71**2}
    first_try = np.array((0, 0, 17.71))

    # 计算折射点
    minimize_res = np.apply_along_axis(lambda v: minimize(gen_optical_path_func(v, PMT_coordinate),\
                                      first_try, constraints=con),\
                                      -1, vertices)

    edge_points = np.stack(np.frompyfunc(lambda y: y.x, 1, 1)(minimize_res))
    successes = np.frompyfunc(lambda y: y.success, 1, 1)(minimize_res)

    # 计算每条光路的长度
    distances = distance(edge_points, vertices) + distance(edge_points, PMT_coordinate)

    # 计算入射角，出射角
    normal_vectors = edge_points
    incidence_vectors = edge_points - vertices
    incidence_angles = np.einsum('ij, ij->i', normal_vectors, incidence_vectors) /\
                         np.apply_along_axis(np.linalg.norm, -1, normal_vectors) / np.apply_along_axis(np.linalg.norm, -1, incidence_vectors)
    emergence_angles = np.arcsin(n_LS/n_water * np.sin(incidence_angles))
    # 判断全反射
    max_incidence_angle = np.arcsin(n_water/n_LS)
    can_transmit = (lambda x: x<max_incidence_angle)(incidence_angles)
    successes = successes * can_transmit
    # 计算折射系数
    Rs = np.square(np.sin(emergence_angles - incidence_angles)/np.sin(emergence_angles + incidence_angles))
    Rp = np.square(np.tan(emergence_angles - incidence_angles)/np.tan(emergence_angles + incidence_angles))
    T = 1 - (Rs+Rp)/2

    res = S * successes * T / np.square(distances) / (4*np.pi)
    return res

# 读入几何文件
with h5.File(args.geo, "r") as geo:
    # 只要求模拟17612个PMT
    PMT_list = geo['Geometry'][:17612]

# 输出
with h5.File(args.opt, "w") as opt:
    # 生成顶点
    ParticleTruth, PhotonTruth = generate_events(args.n)
    
    opt['ParticleTruth'] = ParticleTruth

    
    print("TODO: Write opt file")
