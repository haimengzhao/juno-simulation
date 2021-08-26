import argparse

# 处理命令行
parser = argparse.ArgumentParser()
parser.add_argument("-n", dest="n", type=int, help="Number of events")
parser.add_argument("-g", "--geo", dest="geo", type=str, help="Geometry file")
parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
args = parser.parse_args()

import h5py as h5

# 读入几何文件
with h5.File(args.geo, "r") as geo:
    print("TODO: Deal with geometry file")

# 输出
with h5.File(args.opt, "w") as opt:
    # 循环模拟
    for i in range(args.n):
        print("TODO: Event", i)

    print("TODO: Write opt file")
