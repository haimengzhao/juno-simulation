import argparse
import numpy as np

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
def get_PE_probability(x, y, z, phi, theta):
    '''
    描述：计算(x, y, z)处产生的光子到达(phi, theta)处PMT的概率
    输入：x, y, z: 顶点坐标/mm
          phi, theta: PMT坐标
    输出：float64类型的概率
    算法描述：
    1. 计算从(x, y, z)到PMT可能的两条光路：折射/反射+折射
       可能的思路：费马原理？
    2. 计算这两条光路的总长度
    3. 用菲涅尔公式计算假设光子正好沿着这个方向，真的会这样走的概率
    4. 返回3中概率*(求和 PMT的有效截面/光路总长度^2)/4pi立体角
    '''
    raise NotImplementedError

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
