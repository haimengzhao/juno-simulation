import argparse
import numpy as np
import h5py as h5
from event import generate_events
from utils import save_file

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

if __name__ == "__main__":

    # 处理命令行
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="n", type=int, help="Number of events")
    parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
    parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
    args = parser.parse_args()

    # 读入几何文件
    with h5.File(args.geo, "r") as geo:
        # 只要求模拟17612个PMT
        PMT_list = geo['Geometry'][:17612]


    # 生成顶点
    ParticleTruth, PhotonTruth = generate_events(args.n)

    # TODO: 光学
    PETruth = np.zeros(1)

    # TODO: 波形
    Waveform = np.zeros(1)

    # 保存文件
    save_file(args.opt, ParticleTruth, PETruth, Waveform)
