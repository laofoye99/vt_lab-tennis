from models.tennis_tracker.inference import TennisTracker
# from test.inference_time_delay_profiling import TennisTracker as TennisTracker_time_delay_profiling
from utils.shared_data import SharedData
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.player_tracker.person_detector import PersonDetector
import os
import cv2
import matplotlib.pyplot as plt
import time
import json

def main():
    video_stream = VideoStream()
    frame_processor = FrameProcessor()
    tennis_tracker = TennisTracker()
    person_detector = PersonDetector()
    
    # 存储结果的列表
    results = []
    events_record = []
    frame_count = 0
    
    while True:
        frame = video_stream.get_frame()
        
        if frame is None:
            break
            
        original_frame = frame.copy()
        frame_for_processing = frame_processor.process(frame)
        
        # 处理人员检测
        person_result = person_detector.detect(frame_for_processing, original_frame, event_draw=False)
        
        # 处理网球跟踪
        tennis_tracker.process(frame_for_processing, original_frame, event_draw=False)
        
        # 获取最新事件和坐标
        events= tennis_tracker.get_latest_event()
        if events_record == events:
            continue
        else:
            events_record = events
            print(events)
            
        coordinates = tennis_tracker.get_latest_coordinates()
        for coordinate in coordinates:
            print(coordinate)
        
        # 获取速度信息
        speed_calculator = tennis_tracker.speed_calculator
        speed_results = speed_calculator.get_results()
        
        # 解析人员检测结果
        if person_result and len(person_result) >= 9:
            _, x1, y1, x2, y2, x3, y3, x4, y4 = person_result
            
            # 计算人员中心点坐标（标准化到0-1范围）
            far_count_person_x = (x1 + x2) / 2 / tennis_tracker.frame_width
            far_count_person_y = (y1 + y2) / 2 / tennis_tracker.frame_height
            near_count_person_x = (x3 + x4) / 2 / tennis_tracker.frame_width
            near_count_person_y = (y3 + y4) / 2 / tennis_tracker.frame_height
        else:
            far_count_person_x = far_count_person_y = near_count_person_x = near_count_person_y = 0.0
        
        # 处理网球坐标和事件
        if coordinates and len(coordinates) > 0:
            latest_coord = coordinates[-1]
            if len(latest_coord) >= 9:
                # 坐标格式: [frame_number, timestamp, detected, center_x, center_y, x_proj, y_proj, x_standardized, y_standardized, confidence]
                x_standardized = latest_coord[7]  # 标准化x坐标
                y_standardized = latest_coord[8]  # 标准化y坐标
                detected = latest_coord[2]  # 是否检测到球
                
                # 确保坐标值有效
                if not isinstance(x_standardized, (int, float)) or not isinstance(y_standardized, (int, float)):
                    x_standardized = y_standardized = 0.0
                
                # 确定事件类型
                event_type = "none"
                if events:
                    top_bounces, bottom_bounces, top_shots, bottom_shots = events
                    
                    if top_shots:
                        event_type = "hit"
                    elif bottom_shots:
                        event_type = "hit"
                    elif top_bounces:
                        event_type = "bounce"
                    elif bottom_bounces:
                        event_type = "bounce"
                
                # 获取速度
                speed = speed_results.get('speed', 0)
                
                # 创建结果字典
                result_dict = {
                    "x": round(x_standardized, 2) if detected else 0.0,
                    "y": round(y_standardized, 2) if detected else 0.0,
                    "type": event_type,
                    "speed": round(speed, 2),
                    "timestamp": frame_count,
                    "farCountPerson_x": round(far_count_person_x, 2),
                    "farCountPerson_y": round(far_count_person_y, 2),
                    "nearCountPerson_x": round(near_count_person_x, 2),
                    "nearCountPerson_y": round(near_count_person_y, 2)
                }
                
                results.append(result_dict)
        
        # 显示处理后的帧
        cv2.imshow('person detection', original_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # 保存结果到JSON文件
    output_file = 'tennis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_file}")
    print(f"总共处理了 {len(results)} 帧数据")
    
    # 清理资源
    cv2.destroyAllWindows()
    tennis_tracker.cleanup()
    person_detector.cleanup()

if __name__ == '__main__':
    main()
