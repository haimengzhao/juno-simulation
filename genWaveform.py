import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import save_file
import h5py

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
    return (t > 0) * ampli * np.exp(- t / td) * (1 - np.exp(- t / tr))


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


def get_waveform(PETruth, ampli=1000, td=10, tr=5, ratio=1e-2, noisetype='normal'):
    '''
    根据PETruth生成波形

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
    
    # 读取PETruth
    # 为了节省内存不再使用
    # EventID = PETruth['EventID']
    # ChannelID = PETruth['ChannelID']
    # PETime = PETruth['PETime']

    length = len(PETruth)

    # 采样
    t = np.tile(np.arange(0, 1000, 1), (length, 1))

    # 生成Waveform
    if noisetype == 'normal':
        Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth['PETime'].reshape(-1, 1), ampli, td, tr)
    elif noisetype == 'sin':
        Waveform = sin_noise(t, np.pi/1e30, ratio * ampli) + double_exp_model(t - PETruth['PETime'].reshape(-1, 1), ampli, td, tr)
    else:
        print(f'{noisetype} noise not implemented, use normal noise instead!')
        Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth['PETime'].reshape(-1, 1), ampli, td, tr)

    # 拼接Waveform表
    WF = np.array(list(zip(
        PETruth['EventID'], 
        PETruth['ChannelID'], 
        list(map(lambda x: x.reshape(-1), np.split(Waveform, length, axis=0)))
        )), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])

    # 同事件同PMT波形叠加
    WF_pd = pd.DataFrame.from_records(WF.tolist(), columns=['EventID', 'ChannelID', 'Waveform'])
    # 释放内存
    del WF
    WF_pd = WF_pd.groupby(['EventID', 'ChannelID'], as_index=False).agg({'Waveform': np.sum})
    WF = WF_pd.to_records(index=False).astype([('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])
    del WF_pd
    # 返回Waveform表
    return WF


def get_waveform_bychunk(filename, ParticleTruth, PETruth, ampli=1000, td=10, tr=5, ratio=1e-2, noisetype='normal'):
    '''
    根据PETruth生成波形(对于每个Event分步进行,以节省内存)并保存

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

    # 分Event生成
    if len(Eindex) > 1:

        # 采样
        length = Eindex[1] - Eindex[0]
        t = np.tile(np.arange(0, 1000, 1), (length, 1))

        # 生成Waveform
        if noisetype == 'normal':
            Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[0]:Eindex[1]]['PETime'].reshape(-1, 1), ampli, td, tr)
        elif noisetype == 'sin':
            Waveform = sin_noise(t, np.pi/1e30, ratio * ampli) + double_exp_model(t - PETruth[Eindex[0]:Eindex[1]].reshape(-1, 1), ampli, td, tr)
        else:
            print(f'{noisetype} noise not implemented, use normal noise instead!')
            Waveform = normal_noise(t, ratio * ampli) + double_exp_model(t - PETruth[Eindex[0]:Eindex[1]]['PETime'].reshape(-1, 1), ampli, td, tr)

        # numpy groupby
        # ref:
        # https://stackoverflow.com/questions/58546957/sum-of-rows-based-on-index-with-numpy
        Channels, idx = np.unique(PETruth[Eindex[0]:Eindex[1]]['ChannelID'], return_inverse=True)
        order = np.argsort(idx)
        breaks = np.flatnonzero(np.concatenate(([1], np.diff(idx[order]))))
        # 每个Channel波形求和
        result = np.add.reduceat(Waveform[order], breaks, axis=0)
        # 拼接Waveform表
        WF = np.array(list(zip(
                np.ones(len(Channels))*Events[0], 
                Channels, 
                list(map(lambda x: x.reshape(-1), np.split(result, len(Channels), axis=0)))
                )), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])

        # write to file
        with h5py.File(filename, "a") as f:
            ptds = f.create_dataset('ParticleTruth', data=ParticleTruth)
            petds = f.create_dataset('PETruth', data=PETruth)
            wfds = f.create_dataset('Waveform', (len(Channels),), maxshape=(None,), 
                            dtype=np.dtype([('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))]))
            wfds[:] = WF


        for i in tqdm(range(1, len(Eindex)-1)):
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
            result = np.add.reduceat(Waveform[order], breaks, axis=0)
            WF = np.array(list(zip(
                    np.ones(len(Channels))*Events[i], 
                    Channels, 
                    list(map(lambda x: x.reshape(-1), np.split(result, len(Channels), axis=0)))
                    )), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])

            # incremental write
            with h5py.File(filename, "a") as f:
                wfds = f['Waveform']
                wfds.resize(wfds.shape[0] + len(Channels), axis=0)
                wfds[-len(Channels):] = WF

    else:
        # 只有一个Event
        waveform = get_waveform(PETruth, ampli, td, tr, ratio, noisetype)
        EventID = waveform['EventID']
        ChannelID = waveform['ChannelID']
        WF = waveform['Waveform']
        WF = np.array(list(zip(
            EventID, 
            ChannelID, 
            list(map(lambda x: x.reshape(-1), np.split(WF, len(WF), axis=0)))
            )), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])

        save_file(filename, ParticleTruth, PETruth, WF)

