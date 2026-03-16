import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import threading
import queue
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.player_tracker.player_tracker import PersonDetector
from models.tennis_tracker.inference import TennisTracker

class ParallelProcessor:
    def __init__(self):
        # 初始化视频流
        self.video_stream = VideoStream()
        
        # 初始化帧处理器
        self.frame_processor = FrameProcessor()
        
        # 初始化跟踪器
        self.player_tracker = PersonDetector()
        self.tennis_tracker = TennisTracker()
        
        # 创建处理队列
        self.frame_queue = queue.Queue(maxsize=30)
        self.processed_queue = queue.Queue(maxsize=30)
        
        # # 设置ROI掩码
        # self.frame_processor.load_mask()
        # self.player_tracker.set_roi_mask(self.frame_processor.court_mask)
        
    def process_frame(self, frame):
        # 处理帧
        processed_frame = self.frame_processor.process(frame)
        
        # 创建线程进行并行处理
        player_thread = threading.Thread(
            target=lambda: self.player_tracker.detect(processed_frame.copy(), event_draw=True)
        )
        tennis_thread = threading.Thread(
            target=lambda: self.tennis_tracker.process(processed_frame.copy(), event_draw=True)
        )
        
        # 启动线程
        player_thread.start()
        tennis_thread.start()
        
        # 等待线程完成
        player_thread.join()
        tennis_thread.join()
        
        return processed_frame
        
    def run(self):
        try:
            while True:
                # 获取帧
                frame = self.video_stream.get_frame()
                if frame is None:
                    break
                    
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow('Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.player_tracker.cleanup()
        self.tennis_tracker.cleanup()
        cv2.destroyAllWindows()

def main():
    processor = ParallelProcessor()
    processor.run()

if __name__ == "__main__":
    main() 