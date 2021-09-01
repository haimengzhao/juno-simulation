import numpy as np

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


def noise(t, period=np.pi/10, ampli=1e-2):
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


def get_waveform(PETruth, ampli=1000, td=10, tr=5, period=np.pi/10, ratio=1e-2):
    '''
    根据PETruth生成波形

    输入: PETruth (Structured Array) [EventID, ChannelID, PETime]

    参数:
    ampli, 波形高度;
    td, 整体衰减时间;
    tr, 峰值位置;
    period, 噪声周期;
    ratio, 噪声振幅/波形高度;

    返回:
    Waveform (Structured Array) [EventID, ChannelID, Waveform]
    '''
    
    # 读取PETruth
    EventID = PETruth['EventID']
    ChannelID = PETruth['ChannelID']
    PETime = PETruth['PETime']

    length = len(PETime)

    # 采样
    t = np.tile(np.arange(0, 1000, 1), (length, 1))
    Waveform = noise(t, period, ratio * ampli) + double_exp_model(t - PETime.reshape(-1, 1), ampli, td, tr)

    #返回Waveform表
    return np.array(list(zip(
        EventID, 
        ChannelID, 
        list(map(lambda x: x.reshape(-1), np.split(Waveform, length, axis=0)))
        )), dtype=[('EventID', '<i4'), ('ChannelID', '<i4'), ('Waveform', '<i2', (1000,))])


