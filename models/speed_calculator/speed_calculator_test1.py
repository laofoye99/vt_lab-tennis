import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from enum import Enum

from utils.shared_data import SharedData

class Direction(Enum):
    """球的移动方向"""
    UNKNOWN = 0
    FROM_UPPER = 1  # 从上半场来球
    FROM_LOWER = 2  # 从下半场来球

class SpeedCalculator(SharedData):
    """
    网球速度计算器
    
    通过检测轨迹点在规定区域内的运动计算球速
    """
    
    def __init__(self, min_calculation_interval=0.12):
        super().__init__()
        """
        初始化速度计算器
        
        参数:
            homography_matrix: 单应性矩阵（从真实坐标到图像坐标），可选
            fps: 视频帧率
            y_upper: 上边界Y坐标，默认234（标准场地坐标系）
            y_lower: 下边界Y坐标，默认486（标准场地坐标系）
            x_left: 左边界X坐标，默认99（标准场地坐标系）
            x_right: 右边界X坐标，默认261（标准场地坐标系）
            court_width_std: 标准网球场宽度(米)
            court_height_std: 标准网球场长度(米)
            court_width_pixel: 图像中网球场宽度(像素)
            court_height_pixel: 图像中网球场长度(像素)
            real_points: 源图像中的点坐标，用于计算单应性矩阵，可选
            reference_points: 目标图像中的点坐标，用于计算单应性矩阵，可选
            min_calculation_interval: 两次速度计算之间的最小间隔(秒)，默认0.5秒
        """
        self.min_calculation_interval = min_calculation_interval  # 两次计算之间的最小间隔(秒)
        self.frame_interval = int(min_calculation_interval / self.fps)  # 最小计算间隔的帧数
        self.last_calculation_frame = 0  # 上次计算速度的帧序号
        self.y_adjust = 20
        
        # 计算像素到实际距离的转换比例
        self.scale_x = self.court_width_std / self.court_width_pixel  # m/pixel
        self.scale_y = self.court_height_std / self.court_height_pixel  # m/pixel
        
        # 用于累积轨迹的缓冲区
        self.frame_number_buffer = []
        self.std_trajectory_buffer = []  # 标准场地坐标系中的轨迹点
        self.total_displacement = 0.0  # 累积的总位移
        self.prev_x = None  # 前一个点的X坐标
        self.prev_y = None  # 前一个点的Y坐标
        self.start_frame = None  # 开始累积的帧号
        
        # 最近计算的速度和方向
        self.latest_speed = 0.0
        self.latest_direction = Direction.UNKNOWN
        self.current_direction = Direction.UNKNOWN  # 当前轨迹的移动方向
        
        # 标记是否正在累积
        self.accumulating = False
        
        # 最大球速限制(m/s)
        self.max_speed = 50.0
    
    def reset(self):
        """重置计算器状态"""
        self.frame_number_buffer = []
        self.std_trajectory_buffer = []
        self.total_displacement = 0.0
        self.prev_x = None
        self.prev_y = None
        self.start_frame = None
        self.current_direction = Direction.UNKNOWN
        self.accumulating = False
    
    def project_to_standard_court(self, point):
        """
        将图像坐标点投影到标准场地坐标系
        
        参数:
            point: (x, y) 图像坐标点
            
        返回:
            (x, y) 标准场地坐标系中的点(像素)
        """
        
        # 应用单应性矩阵H进行投影
        transformed = self.apply_homography([point])
        tx, ty = transformed[0]
        return tx, ty
    
    def apply_homography(self, points):
        transformed_points = []
        for point in points:
            homogeneous_point = np.array([point[0], point[1], 1]).reshape(3, 1)
            transformed_homogeneous_point = np.dot(self.homography_matrix, homogeneous_point)
            transformed_point = transformed_homogeneous_point / transformed_homogeneous_point[2]
            transformed_points.append(transformed_point[:2])
        return transformed_points
    
    def transform_to_court_coords(self, points_list):
        """
        将标准场地坐标系距离(像素)中的连续两点转换为实际距离(米)
        
        参数:
            points_list: 标准场地坐标系中的点列表
            
        返回:
            位移(米)
        """
        if len(points_list) < 2:
            return 0.0
        
        # 取最后两个点计算位移
        p1 = points_list[-2]
        p2 = points_list[-1]
        
        # 计算像素位移
        diff_x = p2[0] - p1[0]
        diff_y = p2[1] - p1[1]
        
        # 转换为实际距离(米)
        real_x = diff_x * self.scale_x
        real_y = diff_y * self.scale_y
        
        # 计算位移
        displacement = np.sqrt(real_x**2 + real_y**2)
        
        return displacement
    
    def determine_direction(self, points_list):
        """
        根据最近的点确定移动方向
        
        参数:
            points_list: 标准场地坐标系中的点列表
            
        返回:
            方向枚举
        """
        if len(points_list) < 2:
            return Direction.UNKNOWN
        
        # 取最后两个点确定方向
        p1 = points_list[-2]
        p2 = points_list[-1]
        
        # 根据y值的变化确定方向
        if p2[1] > p1[1]:
            return Direction.FROM_UPPER
        else:
            return Direction.FROM_LOWER
    
    def calculate_speed(self, x_proj, y_proj, frame_number):
        """
        根据当前点计算速度
        
        参数:
            x: 当前点的X坐标
            y: 当前点的Y坐标
            frame_number: 当前帧号
            
        返回:
            (speed, direction) 计算的速度(m/s)和方向
        """
        # 判断点是否在指定区域内
        in_tracking_area = (self.y_upper <= y_proj <= self.y_lower) and (self.x_left <= x_proj <= self.x_right)
        
        if in_tracking_area:
            # 如果在区域内且不在累积状态，则开始累积
            if not self.accumulating:
                self.reset()
                self.accumulating = True
                self.start_frame = frame_number
                self.prev_x, self.prev_y = x_proj, y_proj
                self.total_displacement = 0
                self.current_direction = Direction.UNKNOWN
                
                return self.latest_speed, self.latest_direction
            else:
                # 计算位移
                displacement = np.sqrt(((x_proj - self.prev_x)*self.scale_x)**2 + 
                                      ((y_proj - self.prev_y)*self.scale_y)**2)
                self.total_displacement += displacement
                
                # 计算当前方向
                if y_proj > self.prev_y:
                    self.current_direction = Direction.FROM_UPPER
                else:
                    self.current_direction = Direction.FROM_LOWER
                    
                # 更新前一个点
                self.prev_x, self.prev_y = x_proj, y_proj
                
                return self.latest_speed, self.latest_direction
        else:
            # 点不在跟踪区域内
            if self.accumulating:
                if frame_number - self.last_calculation_frame > self.frame_interval:
                    # 结束累积并计算速度
                    frame_elapsed = frame_number - self.start_frame
                    
                    # 计算时间间隔 (秒)
                    time_elapsed = frame_elapsed / self.fps
                    
                    # 确保时间间隔大于0且有位移
                    if time_elapsed > 0 and self.total_displacement > 0:
                        # 计算速度
                        speed = self.total_displacement / time_elapsed
                        
                        # 限制最大速度
                        if speed > self.max_speed:
                            print(f"速度超过最大限制: {speed:.2f} m/s")
                            speed = self.max_speed
                        
                        # 更新结果
                        self.last_calculation_frame = frame_number
                        self.latest_speed = speed
                        self.latest_direction = self.current_direction
                        
                        # print(f"计算速度: {speed:.2f} m/s, 方向: {self.current_direction}, 位移: {self.total_displacement:.2f}m, 时间: {time_elapsed:.2f}s")
                        
                        # 更新最后计算时间
                        self.last_calculation_frame = frame_number
                else:
                    # 重置跟踪状态
                    self.reset()
                
                return self.latest_speed, self.latest_direction
            
            else:
                # 不在累积状态，直接返回最近结果
                return self.latest_speed, self.latest_direction
            

# if __name__ == "__main__":
#     # 示例用法
#     import os
    
#     input_csv = "/home/max/Videos/Screencasts/annotations/trajectory.csv"
#     output_csv = "/home/max/Videos/Screencasts/annotations/speed_trajectory.csv"
    
#     # 创建速度计算器
#     calculator = SpeedCalculator(fps=25)
    
#     # 测试轨迹数据
#     df = pd.read_csv(input_csv)
#     for index, row in df.iterrows():
#         calculator.calculate_speed(row['x'], row['y'], row['frame_id'])
#         print(f"计算速度: {calculator.latest_speed:.2f} m/s, 方向: {calculator.latest_direction}")

# x_transfer = (x-72)/(288-72)
# y_transfer = (y-126)/(594-126)

# print(x_transfer, y_transfer)
