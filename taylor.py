

import matplotlib.pyplot as plt

class TaylorDiagram:
    """泰勒图类"""
    
    def __init__(self, refstd: float, fig=None, rect=111,
                 label='_', srange=(0, 1.5), extend=False):
        """
        初始化泰勒图
        
        参数:
        ----
        refstd : float
            参考标准差
        fig : matplotlib.figure.Figure, optional
            图形对象
        rect : int
            子图位置
        label : str
            参考标签
        srange : tuple
            标准差范围
        extend : bool
            是否扩展到负相关
        """
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        
        self.refstd = refstd
        tr = PolarAxes.PolarTransform()
        
        # 相关系数标签
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            self.tmax = np.pi/2
        
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd
        
        ghelper = FA.GridHelperCurveLinear(
            tr, extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)
        
        if fig is None:
            fig = plt.figure()
        
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        
        # 调整坐标轴
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].set_axis_direction("top" if extend else "left")
        
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)
        
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)
        
        # 添加参考点
        self.ax.plot([0], self.refstd, 'r*', ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'r--', label='_')
        
        self.samplePoints = []
    
    def add_sample(self, stddev: float, corrcoef: float, *args, **kwargs):
        """添加样本点"""
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l
    
    def add_contours(self, levels=5, **kwargs):
        """添加RMS等值线"""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                            np.linspace(0, self.tmax))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours
