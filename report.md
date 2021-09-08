# JUNO模拟与分析实验报告

**Team: PMTSmasher**

**Members: 沈若寒 刘明昊 赵海萌**

**小组主要分工：**

- 沈若寒：光学过程
- 刘明昊：顶点模拟与光子生成
- 赵海萌：绘图与波形生成

## 目录

[TOC]

## 整体思路

Lmh

## 0. 文件结构与执行方式

### 0.1. 文件结构

本项目的文件结构如下：

```
|-- project-1-junosap-pmtsmasher
    |-- geo.h5
    |-- Makefile
    |-- draw.py
    |-- simulate.py
    |-- scripts
        |-- __init__.py
        |-- drawProbe.py
        |-- event.py
        |-- genPETruth.py
        |-- genWaveform.py
        |-- getProbTime.py
        |-- utils.py
    |-- report.md
    |-- docs
    |-- figs
```

其中 ```geo.h5``` 为JUNO的PMT位置数据， ```Makefile``` 文件定义了文件处理Pipeline， ```simulate.py``` 和 ```draw.py``` 分别完成模拟与绘图的功能， ```scripts``` 文件夹下的各文件完成各个子功能， ```report.md``` 和```figs``` 文件夹为本实验报告及其所用到的样图， ```docs``` 文件夹下存储了本项目的要求。

### 0.2. 执行方式

在项目目录下用 ```shell``` 执行代码

```shell
make
```

可以完整地执行整个项目的流程，生成模拟数据文件 ```data.h5``` 并根据该数据绘制图像 ```figures.pdf``` 。

执行代码

```shell
make data.h5
make figures.pdf
```

可分别生成模拟数据 ```data.h5``` 和绘图 ```figures.pdf``` 。

执行代码

```shell
make clean
```

可以清理生成的 ```data.h5``` 和 ```figures.pdf``` 文件。

若要单独测试 ```simulate.py``` 和 ```draw.py``` ，可以执行

```shell
python3 simulate.py -n <num_of_events> -g geo.h5 -o <output_file> [-p <num_of_pmts>]
python3 draw.py <data_file> -g geo.h5 -o <output_file>
```

例如：

```shell
python3 simulate.py -n 4000 -g geo.h5 -o data.h5
python3 draw.py data.h5 -g geo.h5 -o figures.pdf
```



## 1. 顶点模拟与光子生成

Lmh

### 1.1. 思路

此为模板，可任意修改！

### 1.2. 算法



### 1.3. 主要实现方式



### 1.4. 遇到的问题与解决办法



## 2. 光学过程

srh

## 3. 绘图

### 3.1. 思路

**绘图** 是相对较为独立的一项功能，且具有与其余代码交叉验证的功能，因此我们在项目初期建立了 *Git Branch* **draw** 完成了该功能的初步实现。在使用 *GhostHunter 2021* 的数据集简单检验了代码的正确性后，我们将其 *Merge* 入 **master** ，并应用于辅助验证及调试项目其余功能的实现情况。在顶点模拟与光学过程的功能完成后，我们利用完整生成的data.h5对绘图功能进行了进一步地调整和优化：重构了代码结构，调整了图表外观，利用插值增加了图标的分辨率，使其更易阅读。

具体来说，绘图功能包括三个部分：

1. **函数 ```draw_vertices_density```** : 绘制顶点体密度随半径分布的图像，用于验证所生成的顶点是否符合均匀体密度分布的要求；

2. **函数 ```draw_pe_hit_time```** : 绘制光电子接收时间的直方图，用于展示电子在液闪中激发的光子传播到PMT上所产生的光电子在时间上的分布情况；

3. **函数 ```draw_probe```** : 绘制每个 PMT 对探测器内任意⼀点处顶点激发所产生的光子中，被该PMT接收到的光⼦产⽣的 PE 的电荷数期望（以后简称 *Probe* 函数）的极坐标热力图，用于展示光学过程对光子接受概率分布的影响。

   原则上Probe函数有两种绘制方式：独立于模拟、只利用 **PETruth** 表绘制（以后简称为**Data-driven的Probe图**）和正向模拟光子来绘制（以后简称为**Sim-driven的Probe图**）。我们分别实现了两种绘制方式，并相互交叉验证，为我们程序的可靠性提供了进一步的支持。

### 3.2. 主要实现方式

下面分别对这三个功能的具体实现方式和结果做一个介绍。完整样图请见TODO。

#### 3.2.1. 顶点体密度分布图

顶点体分布图的绘制较为简单：从 **ParticleTruth** 表中读出各顶点的直角坐标 $(x, y, z)$ ，计算得到其与原点的距离  $r = \sqrt{x^2+y^2+z^2}$ ，再利用绘制直方图的函数 ```plt.hist``` 得到顶点数随半径的分布，最后根据
$$
\rho(r) = \frac{\text{d}N}{\text{d}V} = \frac{\text{d}N}{\frac{4\pi}{3}\text{d}(r^3)}
$$
进行体密度修正，计算得到顶点体密度随半径的分布，并绘制成图。在绘图过程中，为了美观，我们将纵坐标体密度的单位设置成预期的均匀体密度 $\rho_0 = \frac{N}{V} \approx 1.72\times 10^{-10} \text{mm}^{-3}$ ，将横坐标半径的单位设置成液闪球的半径 $R_{LS} = 17.71\text{m}$ ，并在 $\rho/\rho_0 = 1$ 处绘制了参考线，使该图更易阅读，最终得到的样图如下图所示，可以看到生成的顶点在随机误涨落范围内大致复合均匀分布的要求。

<img src="./figs/rho.png" alt="顶点体密度分布图" style="zoom: 25%;" />

**注意！** 在进行体密度修正时，不要使用 $\text{d}V = 4\pi r^2\text{d}r$ 的近似，否则会导致在顶点数较少的地方出现与真实体密度相差甚远的偏差，使用$ \Delta V = V_2-V_1 = \frac{4\pi}{3}r_2^3-\frac{4\pi}{3}r_1^3 $ ，并将 ```plt.hist``` 返回的半径 ```bins``` 取平均后作为横坐标，可以有效避免这一问题。

我们注意到，根据大数定律，顶点数密度的涨落会随着顶点数的增加而趋于消失。为了验证我们程序的正确性，我们使用同样的程序在顶点数为 $10^{8}$ 时进行了模拟，得到的顶点数分布如下图所示。可以看到顶点的体密度分布非常均匀！只是在原点附近仍有较小偏差，这是因为原点附近生成顶点的概率较小，样本较少，故涨落较大，符合我们的预期。

<img src="./figs/rho1e8.png" alt="顶点体密度分布图(1e8)" style="zoom:33%;" />



#### 3.2.2. 光电子接收时间直方图

光电子接收时间直方图的绘制更为简单：只需读入 **PETruth** 表，并根据接收时间 *PETime* 利用函数 ```plt.hist``` 绘制直方图即可。图的纵坐标为相应时间段内接收到的光子个数，横坐标接收时间的单位设置为 $\text{ns}$ ，最终产生的样图如下图所示。

TODO: PETime直方图



#### 3.2.3. Data-driven的Probe函数热力图

Data-driven的Probe函数图像是绘图功能最困难的部分，主要是因为其涉及到了**大量数据的双键值索引**，即对 **PETruth** 表的 *EventID* 与 *ChannelID* 两个键值同时进行索引，并将各组PE数求和。实现这一功能最直接的方法是 ```pandas.DataFrame.groupby``` 但是其效率过低。

为了解决这一难题，我们可以利用各PMT全同的假设，将 **PETruth** 表的 *EventID* 替换为对应的坐标 $(x, y, z)$ ，将 *ChannelID* 替换为对应的角坐标 $(\theta, \phi)$ ，然后计算得到它们的相对投影极坐标，最后根据这一投影极坐标调用 ```plt.hist2d``` 计算二维直方图。 **在几何上这等价于将所有的Event绕PMT所在轴旋转到同一个平面内，再将所有PMT绕垂直于该平面的过原点的轴旋转至同一方向，最后将对应位置的PE数求和。** 需要注意的是，这一算法依赖于数据的三个性质：

1. 各PMT可视作全同；
2. 体系绕PMT旋转对称；
3. 顶点体密度均匀。

预处理步骤的具体算法实现如下：

```python
print('Replacing Event & Channel with xyz & geo')

# replace ChannelID with corresponding geo
geo_Channels_i = np.array(
  [np.where(self.geo['ChannelID']==a)[0][0] for a in Channels])
pet_geo_i = geo_Channels_i[Channels_i]
pet_geo = np.stack([
  self.geo['theta'][pet_geo_i] / 180 * np.pi,
  self.geo['phi'][pet_geo_i] / 180 * np.pi ], -1)

# replace EventID with corresponding xyz
xyz_Event_i = np.array(
  [np.where(self.simtruth['EventID']==a)[0][0] for a in Events])
pet_xyz_i = xyz_Event_i[Events_i]
pet_xyz = np.stack([
  self.simtruth['x'][pet_xyz_i],
  self.simtruth['y'][pet_xyz_i],
  self.simtruth['z'][pet_xyz_i]], -1)

# raplace xyz, geo with polar coordinates
pet_polar = np.stack(
  polar_from_xyz(Ro,
                 pet_geo[:, 0],
                 pet_geo[:, 1],
                 pet_xyz[:, 0],
                 pet_xyz[:, 1],
                 pet_xyz[:, 2]),
  -1)

```

预处理完成后，我们得到了PE数在极坐标上每个 $(\text{d}r, \text{d}\theta)$ 小格上的分布，而我们想要绘制的是每个顶点产生的PE数期望。由于顶点体密度均匀，故其在极坐标上的分布并非均匀，因此需要根据公式
$$
\text{Probe}(r, \theta) = \frac{\text{d}N_{PE}}{\text{d}N}
=\frac{\text{d}N_{PE}}{\text{d}V}\frac{\text{d}V}{\text{d}N}
=\frac{\text{d}N_{PE}}{2\pi r\sin\theta\rho_0\text{d}r\text{d}\theta}
$$
做出修正，最后利用 ```plt.pcolormesh``` 绘制出热力图即可。

为了提升图像的分辨率，我们利用 ```scipy.interpolate.interp2d``` 进行了插值。同时为了更方便与同行比较，我们采取了对数着色，并选用了 **cmap**  *Jet* 。最终的样图如下图所示。完整高清大图请见 [PDF: Data-driven Probe](./figs/probe_data.pdf) 。

TODO: Data-driven的Probe图像

#### 3.3.4. Sim-driven的Probe函数热力图

srh



## 4. 波形生成

### 4.1. 思路



## 5. 优化经验总结



## 6. 加分项目

