import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.shared_data import SharedData


class SpeedCalculator(SharedData):
    def __init__(self):
        super().__init__()
        # 区域边界参数
        self.min_displacement = 5
        
        # 运动状态跟踪
        self.prev_x = None
        self.prev_y = None
        self.in_serve_area = False
        self.accumulated_frames = 0
        self.total_distance = 0.0
        self.trajectory_points = []
        self.small_move_count = 0
        
        # 计算结果存储
        self.latest_speed = 0
        self.latest_direction = None
        
        # 帧率相关参数
        self.frame_rate = self.fps
        self.time_interval = 1.0 / self.fps
        
        self.max_speed = 50

    def _is_in_serve_area(self, x, y):
        """检查坐标是否在发球线区域内"""
        return (self.x_left <= x <= self.x_right) and (self.y_upper <= y <= self.y_lower)

    def _calculate_speed(self):
        """计算并更新最终球速"""
        if self.accumulated_frames >= 2:
            total_time = (self.accumulated_frames - 1) * self.time_interval
            if total_time > 0:
                self.latest_speed = self.total_distance / total_time
                self.latest_speed = min(self.latest_speed, self.max_speed)
                # 转换为km/h（可选）
                # self.latest_speed = self.latest_speed * 3.6

    def _determine_direction(self):
        """根据轨迹点判断来球方向"""
        if len(self.trajectory_points) >= 2:
            first_y = self.trajectory_points[0][1]
            last_y = self.trajectory_points[-1][1]
            self.latest_direction = "FROM_UPPER" if last_y > first_y else "FROM_LOWER"

    def add_coordinate(self, x, y):
        """处理新坐标点"""
        current_in_area = self._is_in_serve_area(x, y)
        
        # 进入区域处理
        if current_in_area and not self.in_serve_area:
            self.in_serve_area = True
            self.accumulated_frames = 1
            self.total_distance = 0.0
            self.trajectory_points = [(x, y)]
            self.small_move_count = 0
            self.prev_x, self.prev_y = x, y
        
        # 在区域内处理
        elif current_in_area and self.in_serve_area:
            # 计算像素位移
            dx_pixel = x - self.prev_x
            dy_pixel = y - self.prev_y
            pixel_displacement = (dx_pixel**2 + dy_pixel**2)**0.5
            
            # 单位换算
            dx_meter = dx_pixel * (self.court_width_std / self.court_width_pixel)
            dy_meter = dy_pixel * (self.court_height_std / self.court_height_pixel)
            self.total_distance += (dx_meter**2 + dy_meter**2)**0.5
            
            # 更新状态
            self.accumulated_frames += 1
            self.prev_x, self.prev_y = x, y
            self.trajectory_points.append((x, y))
            
            # 小位移检测
            if pixel_displacement < self.min_displacement:
                self.small_move_count += 1
                if self.small_move_count >= 2:  # 连续3帧中的2次小位移
                    # 重置累积
                    self.accumulated_frames = 1
                    self.total_distance = 0.0
                    self.trajectory_points = [(x, y)]
                    self.small_move_count = 0
            else:
                self.small_move_count = 0
        
        # 离开区域处理
        elif not current_in_area and self.in_serve_area:
            self._calculate_speed()
            self._determine_direction()
            # 重置状态
            self.in_serve_area = False
            self.accumulated_frames = 0
            self.total_distance = 0.0
            self.trajectory_points = []
            self.small_move_count = 0
            self.prev_x = self.prev_y = None

    def get_results(self):
        """获取最新计算结果"""
        return {
            "speed": round(self.latest_speed, 2) if self.latest_speed else 0,
            "direction": self.latest_direction
        }