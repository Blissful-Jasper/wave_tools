"""
Spectral Analysis Module
========================

热带大气波动的频谱分析工具，实现Wheeler-Kiladis波数-频率谱分析。

主要功能：
---------
1. 对称/反对称分解
2. 波数-频率功率谱计算
3. 背景谱平滑
4. 频谱归一化和可视化

作者: Jianpu
邮箱: xianpuji@hhu.edu.cn
"""

import numpy as np
import xarray as xr
import scipy.signal as signal
from scipy import fft
import matplotlib.pyplot as plt
import os
import time
from typing import List, Optional, Tuple

try:
    import cmaps
    DEFAULT_COLORMAP = cmaps.NCV_blu_red
except ImportError:
    DEFAULT_COLORMAP = 'RdBu_r'


def _show_figure(fig) -> None:
    """Show a figure when the active backend supports it."""
    backend = plt.get_backend().lower()
    if "agg" not in backend or "inline" in backend:
        plt.show()
        return

    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return

    if get_ipython() is not None:
        display(fig)


class SpectralConfig:
    """频谱分析配置参数"""
    
    # 分析参数
    WINDOW_SIZE_DAYS = 96
    WINDOW_SKIP_DAYS = 30
    SAMPLES_PER_DAY = 1
    
    # 滤波参数
    FREQ_CUTOFF = 1.0 / WINDOW_SIZE_DAYS
    
    # 绘图参数
    CONTOUR_LEVELS = np.array([0.0, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 4.0])
    COLORMAP = DEFAULT_COLORMAP
    WAVENUMBER_LIMIT = 15


# ============= 工具函数 =============

def smooth_121(array: np.ndarray) -> np.ndarray:
    """应用1-2-1平滑滤波器"""
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0:
        return array
    ok = np.isfinite(array)
    if not np.any(ok):
        return np.full_like(array, np.nan)
    if not np.all(ok):
        x = np.arange(array.size)
        array = np.interp(x, x[ok], array[ok])
    weight = np.array([1., 2., 1.]) / 4.0
    return np.convolve(np.r_[array[0], array, array[-1]], weight, 'valid')


def remove_annual_cycle(data: xr.DataArray, samples_per_day: float, freq_cutoff: float) -> xr.DataArray:
    """去除年循环信号"""
    n_time, _, _ = data.shape
    
    # 去趋势
    detrended_data = signal.detrend(data, axis=0)
    
    # FFT
    fourier_transform = fft.rfft(detrended_data, axis=0)
    frequencies = fft.rfftfreq(n_time, d=1. / float(samples_per_day))
    
    # 低频滤波
    cutoff_index = np.argwhere(frequencies <= freq_cutoff).max()
    if cutoff_index > 1:
        fourier_transform[1:cutoff_index + 1, ...] = 0.0
    
    # 逆FFT
    filtered_data = fft.irfft(fourier_transform, axis=0, n=n_time)
    
    return xr.DataArray(filtered_data, dims=data.dims, coords=data.coords)


def decompose_symmetric_antisymmetric(data_array: xr.DataArray) -> xr.DataArray:
    """
    对称/反对称分解（完全按照Wheeler-Kiladis原始方法）
    
    注意：
    -----
    - symmetric = 0.5 * (data - flip(data))     # 南北反相
    - antisymmetric = 0.5 * (data + flip(data)) # 南北同相
    - 结果数组：南半球存储symmetric，北半球存储antisymmetric
    
    参数:
    -----
    data_array : xr.DataArray
        输入数据，维度为 (time, lat, lon)
    
    返回:
    -----
    result : xr.DataArray
        处理后的数据，南半球=对称分量，北半球=反对称分量
    """
    lat_dim = data_array.dims.index('lat')
    nlat = data_array.shape[lat_dim]
    
    # 计算对称和反对称分量（原始公式）
    symmetric = 0.5 * (data_array.values - np.flip(data_array.values, axis=lat_dim))
    antisymmetric = 0.5 * (data_array.values + np.flip(data_array.values, axis=lat_dim))
    
    # 转为DataArray
    symmetric = xr.DataArray(symmetric, dims=data_array.dims, coords=data_array.coords)
    antisymmetric = xr.DataArray(antisymmetric, dims=data_array.dims, coords=data_array.coords)
    
    # 组合结果：南半球=对称，北半球=反对称
    result = data_array.copy()
    half = nlat // 2
    
    if nlat % 2 == 0:
        # 偶数纬度
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half:, :] = antisymmetric.values[:, half:, :]
    else:
        # 奇数纬度（包含赤道）
        result.values[:, :half, :] = symmetric.values[:, :half, :]
        result.values[:, half+1:, :] = antisymmetric.values[:, half+1:, :]
        result.values[:, half, :] = symmetric.values[:, half, :]  # 赤道使用对称分量
    
    return result


# ============= 主类 =============

class WKSpectralAnalysis:
    """Wheeler-Kiladis频谱分析类"""
    
    def __init__(self, config: Optional[SpectralConfig] = None):
        """
        初始化频谱分析
        
        参数:
        ----
        config : SpectralConfig, optional
            配置参数对象
        """
        self.config = config or SpectralConfig()
        self.raw_data = None
        self.processed_data = None
        self.power_symmetric = None
        self.power_antisymmetric = None
        self.background = None
        self.frequency = None
        self.wavenumber = None
        
    def load_data(self, 
                  data: Optional[xr.DataArray] = None,
                  data_path: Optional[str] = None,
                  variable: str = 'olr',
                  lat_range: Tuple[float, float] = (-15, 15),
                  time_range: Optional[Tuple[str, str]] = None) -> 'WKSpectralAnalysis':
        """
        加载数据
        
        参数:
        ----
        data : xr.DataArray, optional
            直接提供的数据数组
        data_path : str, optional
            NetCDF文件路径
        variable : str
            变量名
        lat_range : tuple
            纬度范围
        time_range : tuple, optional
            时间范围
            
        返回:
        ----
        self
        """
        if data is not None:
            self.raw_data = data
        elif data_path is not None:
            ds = xr.open_dataset(data_path).sortby('lat')
            self.raw_data = ds[variable].sel(lat=slice(*lat_range))
            if time_range:
                self.raw_data = self.raw_data.sel(time=slice(*time_range))
        else:
            raise ValueError("必须提供data或data_path")
        
        # 确保维度顺序
        self.raw_data = self.raw_data.transpose('time', 'lat', 'lon')
        print(f"数据已加载: {self.raw_data.shape} (time, lat, lon)")
        return self
    
    def preprocess(self) -> 'WKSpectralAnalysis':
        """
        预处理数据：去趋势、去年循环、对称/反对称分解
        
        返回:
        ----
        self
        """
        print("预处理中...")
        start_time = time.time()
        
        # 去趋势
        mean_value = self.raw_data.mean(dim='time')
        detrended = signal.detrend(self.raw_data, axis=0, type='linear')
        detrended = xr.DataArray(detrended, dims=self.raw_data.dims, coords=self.raw_data.coords) + mean_value
        
        # 去年循环
        filtered = remove_annual_cycle(detrended, self.config.SAMPLES_PER_DAY, self.config.FREQ_CUTOFF)
        
        # 对称/反对称分解（原始方法）
        self.processed_data = decompose_symmetric_antisymmetric(filtered)
        
        print(f"预处理完成，耗时 {time.time() - start_time:.1f} 秒")
        print(f"  数据形状: {self.processed_data.shape}")
        print(f"  纬度范围: {self.processed_data.lat.values}")
        return self
    
    def compute_spectrum(self) -> 'WKSpectralAnalysis':
        """
        计算波数-频率功率谱
        
        返回:
        ----
        self
        """
        print("计算功率谱...")
        start_time = time.time()
        
        ntim, nlat, nlon = self.processed_data.shape
        spd = self.config.SAMPLES_PER_DAY
        nDayWin = self.config.WINDOW_SIZE_DAYS
        nDaySkip = self.config.WINDOW_SKIP_DAYS
        nSampWin = nDayWin * spd
        nSampSkip = nDaySkip * spd
        nWindow = int((ntim - nSampWin) / (nSampSkip + nSampWin)) + 1
        
        print(f"窗口大小: {nDayWin}天, 跳跃: {nDaySkip}天, 总窗口数: {nWindow}")
        
        # 累积功率
        sumpower = np.zeros((nSampWin, nlat, nlon))
        ntStrt, ntLast = 0, nSampWin
        
        for nw in range(nWindow):
            data_win = self.processed_data[ntStrt:ntLast, :, :]
            data_win = signal.detrend(data_win, axis=0)
            
            # Tukey窗
            window = signal.windows.tukey(nSampWin, 0.1, True)
            data_win *= window[:, np.newaxis, np.newaxis]
            
            # 2D FFT
            power = fft.fft2(data_win, axes=(0, 2)) / (nlon * nSampWin)
            sumpower += np.abs(power) ** 2
            
            ntStrt = ntLast + nSampSkip
            ntLast = ntStrt + nSampWin
        
        sumpower /= nWindow
        
        # 设置频率和波数轴
        if nlon % 2 == 0:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)[1:]
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, nlon:0:-1]
        else:
            self.wavenumber = fft.fftshift(fft.fftfreq(nlon) * nlon)
            sumpower = fft.fftshift(sumpower, axes=2)[:, :, ::-1]
        
        self.frequency = fft.fftshift(fft.fftfreq(nSampWin, d=1./spd))[nSampWin//2:]
        sumpower = fft.fftshift(sumpower, axes=0)[nSampWin//2:, :, :]
        
        # 分离对称/反对称功率
        power_symmetric = np.array(
            2.0 * sumpower[:, nlat//2:, :].sum(axis=1), copy=True
        )
        power_antisymmetric = np.array(
            2.0 * sumpower[:, :nlat//2, :].sum(axis=1), copy=True
        )
        background = np.array(sumpower.sum(axis=1), copy=True)

        # 屏蔽零频率
        power_symmetric[0, :] = np.nan
        power_antisymmetric[0, :] = np.nan
        background[0, :] = np.nan

        # 转为DataArray
        self.power_symmetric = xr.DataArray(
            power_symmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        self.power_antisymmetric = xr.DataArray(
            power_antisymmetric,
            dims=("frequency", "wavenumber"),
            coords={"wavenumber": self.wavenumber, "frequency": self.frequency}
        )
        
        # 背景谱
        self.background = background
        
        print(f"功率谱计算完成，耗时 {time.time() - start_time:.1f} 秒")
        return self
    
    def smooth_background(self, wave_limit: int = 27) -> 'WKSpectralAnalysis':
        """
        平滑背景谱
        
        参数:
        ----
        wave_limit : int
            平滑的波数限制
            
        返回:
        ----
        self
        """
        print("平滑背景谱...")
        wave_indices = np.where(np.abs(self.wavenumber) <= wave_limit)[0]
        
        for idx, freq in enumerate(self.frequency):
            # 根据频率调整平滑次数
            if freq < 0.1:
                n_smooth = 5
            elif freq < 0.2:
                n_smooth = 10
            elif freq < 0.3:
                n_smooth = 20
            else:
                n_smooth = 40
            
            for _ in range(n_smooth):
                self.background[idx, wave_indices] = smooth_121(self.background[idx, wave_indices])
        
        # 频率方向平滑
        for wn_idx in wave_indices:
            for _ in range(10):
                self.background[:, wn_idx] = smooth_121(self.background[:, wn_idx])
        
        print("背景谱平滑完成")
        return self

    def plot_spectrum(self,
                      max_wn: Optional[int] = None,
                      max_freq: float = 0.5,
                      add_matsuno_lines: bool = True,
                      he: Optional[List[float]] = None,
                      cpd_lines: Optional[List[float]] = None,
                      save_path: Optional[str] = None,
                      cmap: Optional[str] = None,
                      levels: Optional[np.ndarray] = None,
                      show: bool = True,
                      close: bool = False):
        """
        绘制Wheeler-Kiladis归一化频谱图。

        该方法封装plotting.plot_wk_spectrum，直接使用当前analysis对象中
        已计算的对称谱、反对称谱和平滑背景谱。

        参数:
        ----
        max_wn : int, optional
            最大绘图波数，默认使用config.WAVENUMBER_LIMIT
        max_freq : float
            最大绘图频率
        add_matsuno_lines : bool
            是否添加Matsuno理论曲线
        he : list of float, optional
            Matsuno曲线的等效深度，默认[8, 25, 90]
        cpd_lines : list of float, optional
            标注周期线，默认[3, 6, 30]
        save_path : str, optional
            图像保存路径
        cmap : str, optional
            色标，默认使用config.COLORMAP
        levels : np.ndarray, optional
            等值线水平，默认使用config.CONTOUR_LEVELS
        show : bool
            是否调用plt.show()
        close : bool
            是否在绘图/保存后关闭figure，批量绘图时建议设为True

        返回:
        ----
        fig, axes
            Matplotlib图形和坐标轴对象
        """
        missing = [
            name for name in (
                "power_symmetric",
                "power_antisymmetric",
                "background",
                "wavenumber",
                "frequency",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "频谱结果尚未计算或平滑，缺少: "
                + ", ".join(missing)
                + "。请先运行 preprocess(), compute_spectrum(), smooth_background()."
            )

        if max_wn is None:
            max_wn = getattr(self.config, "WAVENUMBER_LIMIT", 15)
        if he is None:
            he = [8, 25, 90]
        if cpd_lines is None:
            cpd_lines = [3, 6, 30]
        if cmap is None:
            cmap = getattr(self.config, "COLORMAP", DEFAULT_COLORMAP)
        if levels is None:
            levels = getattr(self.config, "CONTOUR_LEVELS", None)

        try:
            from .plotting import plot_wk_spectrum
        except ImportError:
            from plotting import plot_wk_spectrum

        return plot_wk_spectrum(
            self.power_symmetric,
            self.power_antisymmetric,
            self.background,
            self.wavenumber,
            self.frequency,
            max_wn=max_wn,
            max_freq=max_freq,
            add_matsuno_lines=add_matsuno_lines,
            he=he,
            cpd_lines=cpd_lines,
            save_path=save_path,
            cmap=cmap,
            levels=levels,
            show=show,
            close=close,
        )

    def _background_dataarray(self) -> xr.DataArray:
        """返回带frequency/wavenumber坐标的背景谱。"""
        if self.background is None or self.frequency is None or self.wavenumber is None:
            raise ValueError("背景谱尚未计算。请先运行 compute_spectrum()。")

        return xr.DataArray(
            self.background,
            dims=("frequency", "wavenumber"),
            coords={"frequency": self.frequency, "wavenumber": self.wavenumber},
        )

    def _plot_power_field(self,
                          data: xr.DataArray,
                          ax,
                          title: str,
                          max_wn: int,
                          max_freq: float,
                          use_log: bool,
                          cmap: str,
                          levels):
        """绘制单个二维功率谱场。"""
        plot_data = data.sel(
            frequency=slice(0.0, max_freq),
            wavenumber=slice(-max_wn, max_wn),
        )

        if use_log:
            plot_data = xr.where(plot_data > 0, np.log10(plot_data), np.nan)
            cbar_label = "log10(power)"
        else:
            cbar_label = "power"

        image = plot_data.plot.contourf(
            ax=ax,
            cmap=cmap,
            levels=levels,
            add_colorbar=False,
            extend="both",
        )
        ax.axvline(0, linestyle="--", color="k", linewidth=0.5)
        ax.set_xlim([-max_wn, max_wn])
        ax.set_ylim([0, max_freq])
        ax.set_title(title)
        ax.set_xlabel("Zonal Wavenumber")
        ax.set_ylabel("Frequency (CPD)")
        return image, cbar_label

    def plot_raw_power(self,
                       component: str = "both",
                       max_wn: Optional[int] = None,
                       max_freq: float = 0.5,
                       use_log: bool = True,
                       cmap: str = "magma",
                       levels=30,
                       save_path: Optional[str] = None,
                       show: bool = True,
                       close: bool = False,
                       dpi: int = 200):
        """
        绘制未除以背景谱的原始WK功率谱。

        参数:
        ----
        component : {"both", "symmetric", "antisymmetric"}
            绘制对称、反对称或两个分量
        max_wn : int, optional
            最大绘图波数，默认使用config.WAVENUMBER_LIMIT
        max_freq : float
            最大绘图频率
        use_log : bool
            是否绘制log10(power)
        cmap : str
            色标
        levels : int or array-like
            等值线水平
        save_path : str, optional
            图像保存路径
        show : bool
            是否调用plt.show()
        close : bool
            是否在绘图/保存后关闭figure，批量绘图时建议设为True
        dpi : int
            图像分辨率

        返回:
        ----
        fig, axes
            Matplotlib图形和坐标轴对象
        """
        if self.power_symmetric is None or self.power_antisymmetric is None:
            raise ValueError("原始功率谱尚未计算。请先运行 compute_spectrum()。")

        if max_wn is None:
            max_wn = getattr(self.config, "WAVENUMBER_LIMIT", 15)

        component = component.lower()
        if component not in {"both", "symmetric", "antisymmetric"}:
            raise ValueError("component must be one of: both, symmetric, antisymmetric")

        if component == "both":
            fig = plt.figure(figsize=(12, 6.2), dpi=dpi)
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.08, 0.1], hspace=0.1, wspace=0.25)
            axes = np.array([
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
            ])
            image, cbar_label = self._plot_power_field(
                self.power_symmetric, axes[0], "Raw Symmetric Power",
                max_wn, max_freq, use_log, cmap, levels,
            )
            self._plot_power_field(
                self.power_antisymmetric, axes[1], "Raw Antisymmetric Power",
                max_wn, max_freq, use_log, cmap, levels,
            )
            spacer_ax = fig.add_subplot(gs[1, :])
            spacer_ax.axis("off")
            cbar_ax = fig.add_subplot(gs[2, :])
            fig.colorbar(
                image,
                cax=cbar_ax,
                orientation="horizontal",
                label=cbar_label,
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 6), dpi=dpi)
            data = self.power_symmetric if component == "symmetric" else self.power_antisymmetric
            title = "Raw Symmetric Power" if component == "symmetric" else "Raw Antisymmetric Power"
            image, cbar_label = self._plot_power_field(
                data, ax, title, max_wn, max_freq, use_log, cmap, levels,
            )
            fig.colorbar(image, ax=ax, orientation="vertical", label=cbar_label,pad = 0.05)
            fig.subplots_adjust(right=0.88)
            axes = ax

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"保存至: {save_path}")

        if show:
            _show_figure(fig)

        if close:
            plt.close(fig)

        return fig, axes

    def plot_background_power(self,
                              max_wn: Optional[int] = None,
                              max_freq: float = 0.5,
                              use_log: bool = True,
                              cmap: str = "magma",
                              levels=30,
                              save_path: Optional[str] = None,
                              show: bool = True,
                              close: bool = False,
                              dpi: int = 200):
        """
        绘制背景功率谱。

        如果已经调用 smooth_background()，这里绘制的是平滑后的背景谱；
        如果只调用 compute_spectrum()，这里绘制的是未平滑背景谱。
        """
        if max_wn is None:
            max_wn = getattr(self.config, "WAVENUMBER_LIMIT", 15)

        background = self._background_dataarray()
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5), dpi=dpi)
        image, cbar_label = self._plot_power_field(
            background, ax, "Background Power",
            max_wn, max_freq, use_log, cmap, levels,
        )
        fig.colorbar(image, ax=ax, orientation="vertical", label=cbar_label)
        fig.subplots_adjust(right=0.88)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"保存至: {save_path}")

        if show:
            _show_figure(fig)

        if close:
            plt.close(fig)

        return fig, ax
    
    def save(self, output_path: str) -> 'WKSpectralAnalysis':
        """
        保存频谱到NetCDF
        
        参数:
        ----
        output_path : str
            输出文件路径
            
        返回:
        ----
        self
        """
        ds = xr.Dataset({
            "power_symmetric": self.power_symmetric,
            "power_antisymmetric": self.power_antisymmetric,
            "background": xr.DataArray(
                self.background,
                dims=("frequency", "wavenumber"),
                coords={"frequency": self.frequency, "wavenumber": self.wavenumber}
            )
        })
        ds.to_netcdf(output_path)
        print(f"频谱已保存至: {output_path}")
        return self


# ============= 便捷函数 =============

def calculate_wk_spectrum(data: xr.DataArray,
                          window_days: int = 96,
                          skip_days: int = 30,
                          output_path: Optional[str] = None) -> Tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    """
    便捷函数：一步完成WK频谱计算
    
    参数:
    ----
    data : xr.DataArray
        输入数据
    window_days : int
        窗口大小（天）
    skip_days : int
        窗口跳跃（天）
    output_path : str, optional
        保存路径
        
    返回:
    ----
    power_symmetric, power_antisymmetric, background
    """
    config = SpectralConfig()
    config.WINDOW_SIZE_DAYS = window_days
    config.WINDOW_SKIP_DAYS = skip_days
    
    analysis = WKSpectralAnalysis(config)
    analysis.load_data(data=data)
    analysis.preprocess()
    analysis.compute_spectrum()
    analysis.smooth_background()
    
    if output_path:
        analysis.save(output_path)
    
    return analysis.power_symmetric, analysis.power_antisymmetric, analysis.background
