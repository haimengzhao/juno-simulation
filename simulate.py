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
from tqdm import tqdm
from event import generate_events
from get_prob_time import get_PE_probability, get_random_PE_time
from genWaveform import get_waveform  
from utils import save_file

PMT_COUNT = 17612

if __name__ == "__main__":

    rng = np.random.default_rng()
    # 处理命令行
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", dest="n", type=int, help="Number of events")
    parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
    parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
    args = parser.parse_args()

    # 读入几何文件
    with h5.File(args.geo, "r") as geo:
        # 只要求模拟17612个PMT
        PMT_list = geo['Geometry'][:PMT_COUNT]


    # 生成顶点
    ParticleTruth, PhotonTruth = generate_events(args.n)

    # 光学过程
    PE_prob_cumsum = np.zeros(arg.n)

    for event in tqdm(ParticleTruth):
        for PMT in PMT_list:
            PE_prob_array = np.zeros(17612)
            PE_prob_array[PMT['ChannelID']] = get_PE_probability(
                event['x'], event['y'], event['z'],
                PMT['phi']/180*np.pi, PMT['theta']/180*np.pi
            )
            PE_prob_cumsum[event['EventID']] = np.cumsum(PE_prob_array)
    
    PE_event_ids = np.zeros(PhotonTruth.shape[0])
    PE_channel_ids = np.zeros(PhotonTruth.shape[0])
    PE_petimes = np.zeros(PhotonTruth.shape[0])
    
    index = 0
    for photon in tqdm(PhotonTruth):
        channel_hit = np.asarray(rng.random() < PE_prob_cumsum[photon['EventID']]).nonzero()
        if channel_hit.shape[0] > 0:
            PE_event_ids[index] = photon['EventID']
            PE_channel_ids[index] = channel_hit[0]
            PE_petimes[index] = photon['GenTime'] + get_random_PE_time(
                ParticleTruth[photon['EventID']]['x'],
                ParticleTruth[photon['EventID']]['y'],
                ParticleTruth[photon['EventID']]['z'],
                PMT_list[channel_hit[0]]['phi']/180*np.pi,
                PMT_list[channel_hit[0]]['theta']/180*np.pi
            )
            index += 1

    pe_tr_dtype = [
        ('EventID', '<i4'),
        ('ChannelID', '<i4'),
        ('PETime', '<f8')
    ]
    PETruth = np.zeros(index, dtype=pe_tr_dtype)
    PETruth['EventID'] = PE_event_ids
    PETruth['ChannelID'] = PE_channel_ids
    PETruth['PETime'] = PE_petimes

    # 波形
    Waveform = get_waveform(PETruth, ampli=1000, td=10, tr=5, ratio=0.01, noisetype='normal')

    # 保存文件
    save_file(args.opt, ParticleTruth, PETruth, Waveform)
