import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import h5py
import numexpr as ne

'''
电子学任务
根据PETruth生成波形
'''

def double_exp_model(t, ampli=1000, td=10, tr=5):
    '''
    双指数模型 f(t; ampli, td, tr)

    输入: t, 时间;

    参数:
    ampli=1000, 波形高度;
    td=10ns, 控制整体衰减时间;
    tr=5ns, 控制峰值位置;

    返回:
    f(t) = ampli*exp(-t/td)*(1-exp(-t/tr)), t > 0
           0,                               t <= 0
    '''
    return ne.evaluate('(t > 0) * ampli * exp(- t / td) * (1 - exp(- t / tr))')


def sin_noise(t, period=np.pi/1e30, ampli=1e-2):
    '''
    噪声 noise(t; period, ampli)

    输入: t, 时间;

    参数:
    period=pi/10, 周期;
    ampli=1e-2, 振幅;

    返回:
    noise(t) = ampli*sin(2pi/period*t)
    '''
    return ampli * np.sin(2 * np.pi / period * t)


def normal_noise(t, sigma=5):
    '''
    噪声 noise(t; sigma)

    输入: t, 时间;

    参数:
    sigma, 标准差

    返回:
    noise(t) = N(0, sigma^2)
    '''
    return np.random.normal(0, sigma, t.shape)


def get_waveform(filename, PETruth, ampli=1000, td=10, tr=5, ratio=1e-2, noisetype='normal'):
    '''
    根据PETruth单进程生成波形(对于每个Event分步进行,以节省内存)

    输入: PETruth (Structured Array) [EventID, ChannelID, PETime]

    参数:
    ampli, 波形高度;
    td, 整体衰减时间;
    tr, 峰值位置;
    ratio, 噪声振幅/波形高度;
    noisetype, 噪声形式: 'normal' 正态分布噪声, 'sin' 正弦噪声; 

    返回:
    Waveform (Structured Array) [EventID, ChannelID, Waveform]
    '''
    
    # Event切割
    EventID = PETruth['EventID']

    Events, Eindex = np.unique(EventID, return_index=True)
    Eindex = np.hstack([Eindex, np.array(len(EventID))])

    init = True

    f =  h5py.File(filename, "a")

    # 读取PETruth
    for i in tqdm(range(len(Eindex)-1)):
        # 采样
        length = Eindex[i+1] - Eindex[i]
        t = np.tile(np.arange(0, 1000, 1), (length, 1))

        # 生成Waveform
        if noisetype == 'normal':
            Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)
        elif noisetype == 'sin':
            Waveform = sin_noise(t, np.pi/1e30, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)
        else:
            print(f'{noisetype} noise not implemented, use normal noise instead!')
            Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)

        # numpy groupby
        # ref:
        # https://stackoverflow.com/questions/58546957/sum-of-rows-based-on-index-with-numpy
        Channels, idx = np.unique(PETruth[Eindex[i]:Eindex[i+1]]['ChannelID'], return_inverse=True)
        order = np.argsort(idx)
        breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx[order]))))
        # 同Channel波形相加
        result = np.add.reduceat(Waveform[order], breaks, axis=0)

        # 拼接WF表
        WF = np.empty(len(Channels), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])
        WF['EventID'] = np.ones(len(Channels))*Events[i]
        WF['ChannelID'] = Channels
        WF['Waveform'] = result

        if init:
            # 初始化Waveform表
            wfds = f.create_dataset('Waveform', (len(WF),), maxshape=(None,), 
                    dtype=np.dtype([('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))]))
            init = False
        else:
            # 扩大Waveform表
            wfds.resize(wfds.shape[0] + len(WF), axis=0)

        # 写入WF
        wfds[-len(WF):] = WF

    f.close()




def get_waveform_bychunk(filename, PETruth, ampli=1000, td=10, tr=5, ratio=1e-2, noisetype='normal'):
    '''
    根据PETruth多进程生成波形(对于每个Event分步进行,以节省内存)并保存

    输入: 
    filename, 保存文件名;
    ParticleTruth; PETruth;
    PETruth (Structured Array) [EventID, ChannelID, PETime];

    参数:
    ampli, 波形高度;
    td, 整体衰减时间;
    tr, 峰值位置;
    ratio, 噪声振幅/波形高度;
    noisetype, 噪声形式: 'normal' 正态分布噪声, 'sin' 正弦噪声; 

    返回:
    无
    '''

    # Event切割
    EventID = PETruth['EventID']

    Events, Eindex = np.unique(EventID, return_index=True)
    Eindex = np.hstack([Eindex, np.array(len(EventID))])

    # 分Event并行化生成Waveform
    # ref:
    # https://stackoverflow.com/questions/15704010/write-data-to-hdf-file-using-multiprocessing
    num_processes = 8
    sentinal = None
    write_chunk = 50
    pbar_write = tqdm(total=(len(Eindex)-1)/write_chunk)
    pbar_getwf = tqdm(total=(len(Eindex)-1))

    def getWF_mp(inqueue, output):
        for i in iter(inqueue.get, sentinal):
            # 采样
            length = Eindex[i+1] - Eindex[i]
            t = np.tile(np.arange(0, 1000, 1), (length, 1))

            # 生成Waveform
            if noisetype == 'normal':
                Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)
            elif noisetype == 'sin':
                Waveform = sin_noise(t, np.pi/1e30, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)
            else:
                print(f'{noisetype} noise not implemented, use normal noise instead!')
                Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[i]:Eindex[i+1]]['PETime'].reshape(-1, 1), ampli, td, tr)

            # numpy groupby
            # ref:
            # https://stackoverflow.com/questions/58546957/sum-of-rows-based-on-index-with-numpy
            Channels, idx = np.unique(PETruth[Eindex[i]:Eindex[i+1]]['ChannelID'], return_inverse=True)
            order = np.argsort(idx)
            breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx[order]))))
            # 同Channel波形相加
            result = np.add.reduceat(Waveform[order], breaks, axis=0)

            # 拼接WF表
            WF = np.empty(len(Channels), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])
            WF['EventID'] = np.ones(len(Channels))*Events[i]
            WF['ChannelID'] = Channels
            WF['Waveform'] = result

            # 放入output队列等待写入文件
            pbar_getwf.update()
            output.put([1, WF], block=False)


    def write_file(output):
        # open file
        f =  h5py.File(filename, "a")
        # 初始化Waveform表的Flag
        init = True
        # 多Event同时写入
        ev = write_chunk

        while True:
            # 获取待写入WF
            getoutput = output.get()
            flag = getoutput[0]
            if flag:
                if ev == write_chunk:
                    WF = getoutput[1]
                else:
                    WF = np.concatenate((WF, getoutput[1]), axis=0)
                ev -= 1
                if ev == 0:
                    if init:
                        # 初始化Waveform表
                        wfds = f.create_dataset('Waveform', (len(WF),), maxshape=(None,), 
                                dtype=np.dtype([('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))]))
                        init = False
                    else:
                        # 扩大Waveform表
                        wfds.resize(wfds.shape[0] + len(WF), axis=0)

                    # 写入WF
                    wfds[-len(WF):] = WF
                    pbar_write.update()
                    ev = write_chunk
            else:
                # 写入完成
                if len(WF) > 0:
                    wfds.resize(wfds.shape[0] + len(WF), axis=0)
                    wfds[-len(WF):] = WF
                    pbar_write.update()
                break

        # 关闭文件
        f.close()

    # 待写入文件队列
    output = mp.Queue()
    # 待执行任务队列
    inqueue = mp.Queue()
    # 进程表
    jobs = []
    # 写入文件进程
    proc = mp.Process(target=write_file, args=(output, ))
    proc.start()

    for i in range(num_processes):
        # 生成WF进程
        p = mp.Process(target=getWF_mp, args=(inqueue, output))
        jobs.append(p)
        p.start()

    for i in range(len(Eindex)-1):
        # 分配任务
        inqueue.put(i)

    for i in range(num_processes):
        # 结束任务
        inqueue.put(sentinal)

    for p in jobs:
        p.join()
    
    # 结束文件写入
    output.put([0, None], block=False)
    proc.join()
    pbar_write.close()
    pbar_getwf.close()

    

