import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter, butter, filtfilt
import pywt

def fill_continuous_trajectory(df, kind='cubic'):
    df_new = df.copy()
    frames = df_new['frame_number'].values
    mask = ~df_new['x'].isna()
    if mask.sum() < 2:
        df_new['center_x'] = df_new['x']
        df_new['center_y'] = df_new['y']
        return df_new

    interp_x = interp1d(frames[mask], df_new.loc[mask, 'x'].values, kind=kind, fill_value='extrapolate')
    interp_y = interp1d(frames[mask], df_new.loc[mask, 'y'].values, kind=kind, fill_value='extrapolate')

    df_new['center_x'] = interp_x(frames)
    df_new['center_y'] = interp_y(frames)
    return df_new

# ---------------------------------------------------------------------------
def fit_quadratic(points):
    """
    输入三个二维点 (x, y)，返回经过这些点的二次函数系数和可调用函数。
    函数形式：y = ax² + bx + c
    """
    if len(points) != 3:
        raise ValueError("需要恰好三个点来定义二次函数")
    
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    if len(np.unique(x)) < 3:
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if x[i] == x[j] and y[i] != y[j]:
                    raise ValueError("重复的 x 值对应不同的 y 值，无法定义函数")
    
    A = np.vander(x, 3)
    B = y
    
    coefficients = np.linalg.solve(A, B)
    a, b, c = coefficients
    
    def quadratic_func(x_val):
        return a * x_val**2 + b * x_val + c
    
    return (a, b, c), quadratic_func

def interpolate_x(data, max_nan=8):
    """
    处理x坐标的线性插值
    """
    arr = np.array(data, dtype=float)
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    nan_mask = np.isnan(x)
    
    segments = []
    start = None
    for i, is_nan in enumerate(nan_mask):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            segments.append((start, i-1))
            start = None
    if start is not None:
        segments.append((start, len(arr)-1))
    
    for seg in reversed(segments):
        start, end = seg
        num_nan = end - start + 1
        
        if num_nan > max_nan:
            continue
        
        prev_idx = start - 1
        next_idx = end + 1
        
        if prev_idx < 0 or next_idx >= len(arr):
            continue
        
        x_prev = x[prev_idx]
        x_next = x[next_idx]
        
        delta = (x_next - x_prev) / (num_nan + 1)
        
        for i in range(num_nan):
            x[start + i] = x_prev + delta * (i + 1)
    
    return x.tolist()

def interpolate_y(data):
    """
    改进版Y坐标插值：同时考虑段前和段后的有效点进行拟合
    """
    arr = np.array(data, dtype=float)
    # x = arr[:, 0].copy()
    # y = arr[:, 1].copy()
    x = arr[:, 0]
    y = arr[:, 1]
    y_nan_mask = np.isnan(y)
    y_segments = []
    current_start = None
    
    # 检测连续NaN段
    for i, yi in enumerate(y):
        if np.isnan(yi) and current_start is None:
            current_start = i
        elif not np.isnan(yi) and current_start is not None:
            y_segments.append((current_start, i-1))
            current_start = None
    if current_start is not None:
        y_segments.append((current_start, len(arr)-1))
    
    for seg in y_segments:
        start, end = seg
        ref_points = []
        
        # 向前查找有效点（最多3个）
        forward_idx = start - 1
        while forward_idx >= 0 and len(ref_points) < 3:
            if not np.isnan(y[forward_idx]):
                ref_points.append((x[forward_idx], y[forward_idx]))
            forward_idx -= 1
        
        # 向后查找有效点（最多3个）
        backward_idx = end + 1
        while backward_idx < len(y) and len(ref_points) < 6:  # 前3+后3
            if not np.isnan(y[backward_idx]):
                ref_points.append((x[backward_idx], y[backward_idx]))
            backward_idx += 1
        
        if len(ref_points) < 3:
            continue  # 无法进行拟合
        
        # 按x值排序并选择最近的3个点
        ref_points.sort(key=lambda p: p[0])
        ref_points = ref_points[-3:]  # 优先使用靠近缺失段的点
        # print(f"ref_points: {ref_points}")
        try:
            # 使用二次拟合
            (a, b, c), quad_func = fit_quadratic(ref_points)
            for i in range(start, end + 1):
                y[i] = quad_func(x[i])
        except (ValueError, np.linalg.LinAlgError):
            # 线性插值回退
            if len(ref_points) >= 2:
                x_vals = [p[0] for p in ref_points[-2:]]
                y_vals = [p[1] for p in ref_points[-2:]]
                # print(x_vals)
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                for i in range(start, end + 1):
                    y[i] = slope * x[i] + intercept
    
    return y.tolist()

def interpolate(data, max_nan=8):
    """
    组合插值流程
    返回：完整处理后的数据列表
    """
    # 先处理X坐标
    x_filled = interpolate_x(data, max_nan)
    # 生成中间数据
    temp_data = [[x, orig_y] for x, (_, orig_y) in zip(x_filled, data)]
    # 再处理Y坐标
    y_filled = interpolate_y(temp_data)
    # 组合最终结果
    return [[x, y] for x, y in zip(x_filled, y_filled)]

class Trajectory:
    def __init__(self, df, merge_window=5):
        self.df = df.sort_values('frame_number').reset_index(drop=True)
        self._preprocess_data()
        self._interpolate_center()
        self._compute_properties()
        self.bottom_window = merge_window // 2 + 1
        self.upper_window = merge_window // 2 
        self.med_line = 610 

    def _preprocess_data(self):
        """初始化计算（确保数值类型）"""
        # 原始列转为数值
        # coord_cols = ['x1', 'y1', 'x2', 'y2']
        # self.df[coord_cols] = self.df[coord_cols].apply(pd.to_numeric, errors='coerce')
        
        # 计算中心点（数值运算）
        self.df['center_x'] = self.df['x']
        self.df['center_y'] = self.df['y']
    
    def _interpolate_center(self):
        """组合插值流程（修改后的方法）"""
        # 提取原始数据
        raw_data = self.df[['center_x', 'center_y']].values.tolist()
        # 执行组合插值
        interpolated = interpolate(raw_data)
        # 更新DataFrame
        self.df['center_x'] = [p[0] for p in interpolated]
        self.df['center_y'] = [p[1] for p in interpolated]
        # 二次类型转换确保安全
        self.df['center_x'] = pd.to_numeric(self.df['center_x'], errors='coerce')
        self.df['center_y'] = pd.to_numeric(self.df['center_y'], errors='coerce')
    
    def _compute_properties(self):
        """计算运动属性（类型安全版）"""
        # 验证类型
        if self.df['center_x'].dtype != np.float64:
            raise TypeError("center_x 必须为数值列")
        
        # 计算差分
        self.df['velocity_x'] = self.df['center_x'].diff()
        self.df['velocity_y'] = self.df['center_y'].diff()
        self.df['speed'] = np.hypot(
            self.df['velocity_x'], 
            self.df['velocity_y']
        )

    @property
    def centers(self):
        """获取所有中心点坐标"""
        return self.df[['center_x', 'center_y']].values.tolist()
    
    @property
    def areas(self):
        """获取所有框面积"""
        return self.df['box_area'].values.tolist()
    
    @property
    def velocities(self):
        """获取速度向量列表"""
        return self.df[['velocity_x', 'velocity_y']].values.tolist()

    def _angle_between(self, v1, v2):
        """计算两个向量间的角度（单位：度）"""
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(dot) * 180 / np.pi

    def detect_turning_points(self, 
                             angle_threshold=40, 
                             filter_parabola=True):
        """
        改进版拐点检测
        参数：
            angle_threshold: 角度变化阈值（度）
            merge_distance: 合并相邻拐点的最大间距
            filter_parabola: 是否过滤抛物线顶点
        返回：
            拐点信息列表，格式 [(frame_number, 坐标点, 角度变化), ...]
        """
        trajs = self.centers
        raw_points = []

        # 预处理：确保有足够的轨迹点
        if len(trajs) < 3:
            return []

        # 第一步：初步检测所有候选拐点
        for i in range(1, len(trajs)-1):
            # 排除包含NaN值的点
            if any(np.isnan(trajs[i])) or any(np.isnan(trajs[i-1])) or any(np.isnan(trajs[i+1])):
                continue
            
            v1 = np.array(trajs[i]) - np.array(trajs[i-1])
            v2 = np.array(trajs[i+1]) - np.array(trajs[i])
            
            angle = self._angle_between(v1, v2)
            
            if angle > angle_threshold:
                if filter_parabola and self._is_parabola_vertex(i):
                    continue
                raw_points.append((
                    self.df.iloc[i]['frame_number'],
                    trajs[i],
                    angle
                ))


        # 第二步：智能合并相邻拐点
        merged = []
        if not raw_points:
            return merged

        current = raw_points[0]

        for point in raw_points[1:]:
            # 动态判断合并窗口
            current_y = current[1][1]  # 当前拐点的y坐标
            point_y = point[1][1]      # 下一个拐点的y坐标
    
            # 根据半场选择合并窗口（示例值，请根据实际场地调整）
            if current_y < self.med_line and current_y > 0 and point_y < self.med_line and point_y > 0:  # 上半场
                merge_window = self.upper_window  # 例如5帧
            else:                         # 下半场
                merge_window = self.bottom_window  # 例如8帧

            # 判断是否需要合并
            if (point[0] - current[0]) <= merge_window:
                # 合并策略：选择角度更大的拐点
                current = max([current, point], key=lambda x: x[2])
            else:
                merged.append(current)
                current = point
        
        # 添加最后一个拐点
        merged.append(current)

        # 后处理：过滤异常点
        return self._filter_false_positives(merged)
            
    def _filter_false_positives(self, points):
        """过滤误检测的拐点"""
        valid_points = []
        for p in points:
            # 示例过滤条件：至少需要3帧持续运动
            frame_number = p[0]
            idx = self.df[self.df['frame_number'] == frame_number].index[0]
            if len(self.df) - idx < 3:
                continue
            valid_points.append(p)
        return valid_points

    def _is_parabola_vertex(self, idx):
        """改进的抛物线顶点检测"""
        # 检查前后5帧的坐标变化
        x_coords = self.df['center_x'].iloc[max(0, idx-20):min(len(self.df), idx+5)]
        y_coords = self.df['center_y'].iloc[max(0, idx-20):min(len(self.df), idx+5)]
        
        # 使用二次拟合判断曲率
        try:
            coeffs = np.polyfit(x_coords, y_coords, 2)
            return coeffs[0] < -0.001  # 开口向下说明是顶点
        except:
            return False
        
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

class EnhancedTrajectory:
    def __init__(self, df, turning_points):
        """
        参数：
            df: 原始数据，需包含列 ['frame_number', 'x1', 'y1', 'x2', 'y2']
            turning_points: 拐点列表，格式 [(frame_number, [x,y], angle)]
        """
        self.df = df.sort_values('frame_number').reset_index(drop=True)
        self.turning_points = turning_points
        self._preprocess_data()
        self._mark_turning_points()
        self._mark_missing_segments()
        
    def _preprocess_data(self):
        """预处理：计算中心点并标记缺失"""
        self.df['center_x'] = self.df['x']
        self.df['center_y'] = self.df['y']
        # self.df['center_x'] = (self.df['x1'] + self.df['x2']) / 2
        # self.df['center_y'] = (self.df['y1'] + self.df['y2']) / 2
        self.df['is_missing'] = self.df['center_x'].isna()
        
    def _mark_missing_segments(self):
        """标记连续缺失段"""
        self.df['missing_group'] = self.df['is_missing'].astype(int).diff().ne(0).cumsum()

    def _mark_turning_points(self):
        """新增方法：标记拐点位置"""
        # 提取拐点帧号
        turning_frame_numbers = {tp[0] for tp in self.turning_points}
        # 创建新列
        self.df['turning_point'] = self.df['frame_number'].isin(turning_frame_numbers)
        
    def _is_near_turning(self, segment_frames):
        """判断缺失段是否靠近拐点"""
        return any(
            any(abs(f - tp[0]) <= 2 for f in segment_frames)
            for tp in self.turning_points
        )
    
    def _spline_interpolate(self, valid_frames, valid_centers, missing_frames):
        """三次样条插值"""
        cs_x = CubicSpline(valid_frames, valid_centers[:, 0])
        cs_y = CubicSpline(valid_frames, valid_centers[:, 1])
        return cs_x(missing_frames), cs_y(missing_frames)
    
    def _linear_interpolate(self, method='linear'):
        """线性插值（Pandas内置）"""
        self.df['center_x'] = self.df['center_x'].interpolate(method=method)
        self.df['center_y'] = self.df['center_y'].interpolate(method=method)
    
    def adaptive_interpolate(self):
        """执行自适应插值"""
        # 按缺失段分组处理
        groups = self.df.groupby('missing_group', group_keys=False)
        
        # 处理每个缺失段
        self.df = groups.apply(lambda g: 
            self._process_segment(g) if g['is_missing'].any() else g
        )
        
        # 填充残余缺失值
        self._linear_interpolate()
        
        return self
    
    def _process_segment(self, segment):
        """处理单个缺失段"""
        segment_frames = segment['frame_number'].values
        all_frames = self.df['frame_number'].values
        
        # 获取有效参考点（前后各扩展5帧）
        context_start = max(0, np.searchsorted(all_frames, segment_frames[0]) - 5)
        context_end = min(len(self.df), np.searchsorted(all_frames, segment_frames[-1]) + 5)
        context = self.df.iloc[context_start:context_end]
        
        valid = context[~context['is_missing']]
        if len(valid) < 2:
            return segment  # 无法插值
        
        # 根据是否靠近拐点选择插值方法
        if self._is_near_turning(segment_frames):
            # 三次样条插值
            x_new, y_new = self._spline_interpolate(
                valid['frame_number'], 
                valid[['center_x', 'center_y']].values,
                segment['frame_number']
            )
        else:
            # 线性插值
            x_new = np.interp(segment['frame_number'], valid['frame_number'], valid['center_x'])
            y_new = np.interp(segment['frame_number'], valid['frame_number'], valid['center_y'])
        
        # 更新结果
        segment['center_x'] = x_new
        segment['center_y'] = y_new
        return segment
    
    def get_centers(self):
        """返回插值后的完整中心点坐标"""
        return self.df[['center_x', 'center_y']].values.tolist()
    
    def get_full_df(self):
        """获取完整 DataFrame（含拐点标记）"""
        return self.df
