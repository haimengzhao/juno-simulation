import numpy as np
from scipy.integrate import quad
import utils

LS_RADIUS = 17.71 # 液闪的半径，单位m
SIGMA = 5 # 正态分布的标准差
TAU = 20 # 指数衰减函数e^(-t/tau)中的tau
NORM_FACTOR = 69.15044738473783 # 期望的归一化系数
T_MAX = 500 # 只考虑500ns以内产生的光子
PRECISION = 10000 # expectation取样时的间隔为其倒数


def expectation(t):
    '''
    README中的lambda(t)，非齐次泊松过程的期望
    输入：时间t
    输出：期望值
    '''
    return (quad(lambda s: np.exp(-t/TAU-(t/TAU-s)**2/50), 0, np.inf)[0] 
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
    expect_vect = np.vectorize(expectation)
    expect_list = expect_vect(np.arange(0, 500, 1/PRECISION))
    print("初始化完成！")

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
    print("event位置生成完成！")

    # 生成ParticleTruth
    par_tr_dtype = [
        ('EventID', '<i4'),
        ('x', '<f8'),
        ('y', '<f8'),
        ('z', '<f8'),
        ('p', '<f8')
    ]
    Particle_Truth = np.array(list(zip(
        np.arange(number_of_events),
        event_coordinates[:, 0]*1000,
        event_coordinates[:, 1]*1000,
        event_coordinates[:, 2]*1000,
        np.zeros(number_of_events) + 1
    )), dtype=par_tr_dtype)
    print("Particle_Truth表生成完成！")

    # 生成光子，先用齐次泊松分布，再用expectation来thin
    print("生成光子中...")
    photon_counts = np.round(
        rng.poisson(expectation(0)*T_MAX, number_of_events)
        ).astype(int)
    gen_times = []
    for photon_count in photon_counts:
        gen_time = np.sort(rng.random(photon_count) * T_MAX)
        expe_vec = np.vectorize(expectation)
        gen_time = gen_time[
            rng.random(dtype=np.float32) < 
            expect_list[gen_time.astype(int)*PRECISION]/expect_list[0]
        ]
        gen_times.append(gen_time)
    
    print("生成完成！")

    return Particle_Truth, None
    






