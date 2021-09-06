'''
simulate.py: 生成模拟数据，保存在hdf5文件中

必须的参数：
-n: Number of events
-g, --geo: Geometry file
-o, --output: Output file

可选的参数：
-p --pmt: Number of PMTs, default is 17612

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
from tqdm import tqdm
from scripts.event import generate_events
from scripts.genPETruth import get_PE_Truth
from scripts.genWaveform import get_waveform_bychunk  
from scripts.utils import save_file

if __name__ == "__main__":

    rng = np.random.default_rng()
    # 处理命令行
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="n", type=int, help="Number of events")
    parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
    parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
    parser.add_argument("-p", "--pmt", dest="pmt_count", type=int, help="Number of PMTs, default is 17612", default=17612)
    args = parser.parse_args()

    # 读入几何文件
    with h5.File(args.geo, "r") as geo:
        # 只要求模拟17612个PMT
        PMT_list = geo['Geometry'][:args.pmt_count]

    # 生成顶点
    ParticleTruth, PhotonTruth = generate_events(args.n)

    # 光学过程
    PETruth = get_PE_Truth(ParticleTruth, PhotonTruth, PMT_list, args.n)

    # 保存ParticleTruth和PETruth
    save_file(args.opt, ParticleTruth, PETruth)

    # 中断可直接读取
    # with h5.File('data.h5', 'r') as inp:
    #     ParticleTruth = inp['ParticleTruth'][...]
    #     PETruth = inp['PETruth'][...]

    # 生成波形同时保存
    get_waveform_bychunk(args.opt, ParticleTruth, PETruth, ampli=1000, td=10, tr=5, ratio=0.01, noisetype='normal')

