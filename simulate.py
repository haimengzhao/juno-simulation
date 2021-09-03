import argparse
import numpy as np
import h5py as h5
from event import generate_events
from genWaveform import get_waveform  
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

    # TODO: 光学
    PETruth = np.zeros(1)

    # 波形
    Waveform = get_waveform(PETruth, ampli=1000, td=10, tr=5, ratio=0.01, noisetype='normal')

    # 保存文件
    save_file(args.opt, ParticleTruth, PETruth, Waveform)
