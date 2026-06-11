# Wave Tools

中文 | [English](README.en.md)

Wave Tools 是一个用于热带大气波动诊断的 Python 工具集，主要面向
Wheeler-Kiladis 波数-频率谱、赤道波滤波、交叉谱、Matsuno 色散曲线、
EOF、相位合成和常用科研绘图工作流。

这个仓库目前以源码形式维护，适合在分析脚本、Notebook 或模式后处理流程中直接调用。

## 主要功能

- Wheeler-Kiladis 频谱分析，包括对称/反对称分解、背景谱平滑和谱图绘制
- Kelvin、ER、MRG、IG、TD 和 MJO 等波动的频率-波数滤波
- 两个变量之间的交叉谱、相干性和相位关系分析
- Matsuno 浅水波理论模态和色散曲线计算
- EOF/PCA 分析、Kelvin 波相位合成和滞后合成
- 经纬度图、Hovmoller 图、WK 频谱图、泰勒图和矢量图例
- ICON/HEALPix 非结构网格到规则经纬度网格的辅助转换

## 安装

```bash
git clone git@github.com:Blissful-Jasper/wave_tools.git
python -m pip install -r wave_tools/requirements.txt
```

如果从仓库父目录运行 Python，可以直接导入：

```python
import wave_tools

print(wave_tools.get_version())
```

## 快速示例

### Wheeler-Kiladis 频谱

```python
import xarray as xr
from wave_tools.spectral import calculate_wk_spectrum

olr = xr.open_dataset("olr.nc")["olr"]

power_sym, power_asym, background = calculate_wk_spectrum(
    olr,
    window_days=96,
    skip_days=30,
    output_path="wk_spectrum.nc",
)
```

### Kelvin 波滤波

```python
import xarray as xr
from wave_tools.filters import CCKWFilter

pr = xr.open_dataarray("pr_daily.nc")

filt = CCKWFilter(
    ds=pr,
    sel_dict={"lat": slice(-15, 15)},
    wave_name="kelvin",
    units="mm/day",
    spd=1,
    n_workers=4,
    verbose=True,
)

kelvin = filt.process()
kelvin.to_netcdf("pr_kelvin.nc")
```

### 交叉谱分析

```python
import xarray as xr
from wave_tools.cross_spectrum import quick_cross_spectrum

pr = xr.open_dataarray("pr.nc")
olr = xr.open_dataarray("olr.nc")

result = quick_cross_spectrum(
    pr,
    olr,
    segLen=96,
    segOverLap=-65,
    symmetry="symm",
)

coherence_sq = result["STC"].sel(component="COH2")
phase = result["STC"].sel(component="PHAS")
```

### 绘制 WK 频谱

```python
from wave_tools.plotting import plot_wk_spectrum

fig, axes = plot_wk_spectrum(
    power_sym,
    power_asym,
    background,
    wavenumber=power_sym["wavenumber"],
    frequency=power_sym["frequency"],
    add_matsuno_lines=True,
    save_path="wk_spectrum.png",
)
```

## 模块概览

| 模块 | 用途 |
| --- | --- |
| `spectral.py` | Wheeler-Kiladis 频谱分析 |
| `filters.py` | 赤道波滤波和 CCKW 滤波流程 |
| `cross_spectrum.py` | 单次交叉谱、相干性和相位计算 |
| `cross_spectrum_analysis.py` | 多实验交叉谱分析和绘图辅助函数 |
| `matsuno.py` | Matsuno 理论模态和色散曲线 |
| `phase.py` | 峰值检测、Kelvin 波相位和合成分析 |
| `eof.py` | EOF/PCA 分析 |
| `plotting.py` | WK 频谱、地图、泰勒图等绘图函数 |
| `Xianpumap.py` | 西太平洋底图和 Hovmoller 绘图工具 |
| `utils.py` | 数据加载、模型筛选、Radon 诊断和网格转换 |
| `diagnostics.py` | GMS 和热力学诊断 |

## 数据约定

大多数函数默认输入为 `xarray.DataArray`，并假定维度名称包含：

- `time`
- `lat`
- `lon`

滤波和频谱函数通常要求时间采样均匀。使用 `CCKWFilter` 时，`spd` 表示每天采样次数；日数据取 `spd=1`，6 小时数据取 `spd=4`。

## 依赖

核心依赖包括 `numpy`、`xarray`、`scipy`、`matplotlib`、`pandas`、`cartopy`、`numba`、`joblib`、`scikit-image` 和 `healpy`。部分诊断函数还会用到 `metpy` 和 `geocat-comp`。

## 引用

如果在论文或报告中使用本工具包，建议同时引用具体使用的方法文献，例如：

- Wheeler, M. and Kiladis, G. N. (1999). Convectively coupled equatorial waves: Analysis of clouds and temperature in the wavenumber-frequency domain.
- Matsuno, T. (1966). Quasi-geostrophic motions in the equatorial area.

## 维护者

Jianpu
Hohai University
xianpuji@hhu.edu.cn
