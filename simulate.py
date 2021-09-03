'''
simulate.py: 生成模拟数据，保存在hdf5文件中

必须的参数：
-n: Number of events
-g, --geo: Geometry file
-o, --output: Output file

输出格式：
文件名opt，格式为hdf5

ParticleTruth 表:
EventID 事件编号     '<i4'
x       顶点坐标x/mm '<f8'
y       顶点坐标y/mm '<f8'
z       顶点坐标z/mm '<f8'
p       顶点动量/MeV '<f8'

PETruth 表:
EventID   事件编号      '<i4'
ChannelID PMT 编号      '<i4'
PETime    PE击中时间/ns '<f8'

Waveform 表
EventID   事件编号 '<i4'
ChannelID PMT编号  '<i4'
Waveform  波形     '<i2', (1000,)
'''

import argparse
import numpy as np
import h5py as h5
from event import generate_events
from utils import save_file

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

    # 光学过程
    
    PETruth = np.zeros(1)

    # TODO: 波形
    Waveform = np.zeros(1)

    # 保存文件
    save_file(args.opt, ParticleTruth, PETruth, Waveform)
