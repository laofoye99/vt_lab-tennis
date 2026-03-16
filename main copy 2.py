from models.tennis_tracker.inference import TennisTracker
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.player_tracker.player_tracker import PersonDetector
import threading
import queue
import cv2
import time

class Pipeline:
    def __init__(self, max_queue_size=10):
        self.video_stream = VideoStream()
        self.frame_processor = FrameProcessor()
        self.tennis_tracker = TennisTracker()
        self.person_detector = PersonDetector()
        
        self.frames_delay_buffer = []
        self.process_delay_buffer = []
        
        self.raw_queue = queue.Queue(maxsize=max_queue_size)
        self.processed_queue = queue.Queue(maxsize=max_queue_size*2)
        
        self.viz_queues = {
            'original': queue.Queue(maxsize=max_queue_size),
            'tennis': queue.Queue(maxsize=max_queue_size),
            'player': queue.Queue(maxsize=max_queue_size)
        }
        
        self.stream_active = threading.Event()
        self.processing_started = threading.Event()
        
        self.frame_counter = 0
        self.start_time = time.time()
        
    def video_capture_thread(self):
        """视频采集线程（新增原始帧可视化队列投放）"""
        print("Video capture thread started")
        while self.stream_active.is_set():
            frame = self.video_stream.get_frame()
            if frame is not None:
                try:
                    self.raw_queue.put_nowait(frame)
                    # 同时发送到原始视频显示队列
                    self.viz_queues['original'].put_nowait(frame.copy())
                except queue.Full:
                    _ = self.raw_queue.get()
                    self.raw_queue.put_nowait(frame)
            else:
                time.sleep(0.001)
        
    def preprocessing_thread(self):
        """预处理线程"""
        print("Preprocessing thread started")
        while self.processing_started.wait(1) or not self.raw_queue.empty():
            try:
                raw_frame = self.raw_queue.get(timeout=0.1)
                processed_frame = self.frame_processor.process(raw_frame)
                self.processed_queue.put(processed_frame)
                
                self.frame_counter += 1
                if self.frame_counter % 100 == 0:
                    elapsed = time.time() - self.start_time
                    print(f"Processing FPS: {self.frame_counter/elapsed:.2f}")
                
            except queue.Empty:
                continue
            
    def tennis_detection_thread(self):
        """网球检测线程（新增可视化处理）"""
        print("Tennis detection thread started")
        while self.processing_started.is_set():
            try:
                processed_frame = self.processed_queue.get(timeout=0.1)
                
                # 执行检测并绘制结果
                coordinates = self.tennis_tracker.process(processed_frame)
                viz_frame = self._draw_tennis_markers(processed_frame.copy(), coordinates)
                
                # 发送到可视化队列
                self.viz_queues['tennis'].put_nowait(viz_frame)
                
            except queue.Empty:
                continue
        
    def player_detection_thread(self):
        """球员检测线程（新增可视化处理）"""
        print("Player detection thread started")
        while self.processing_started.is_set():
            try:
                processed_frame = self.processed_queue.get(timeout=0.1)
                
                # 执行检测并绘制结果
                results = self.person_detector.detect(processed_frame)
                viz_frame = self._draw_player_boxes(processed_frame.copy(), results)
                
                # 发送到可视化队列
                self.viz_queues['player'].put_nowait(viz_frame)
                
            except queue.Empty:
                continue
    
    def visualization_thread(self):
        """独立可视化线程（新增）"""
        print("Visualization thread started")
        while self.stream_active.is_set() or any(not q.empty() for q in self.viz_queues.values()):
            # 从各队列获取最新帧
            frames = {}
            for stream_type, q in self.viz_queues.items():
                while not q.empty():  # 只保留最新帧
                    frames[stream_type] = q.get()
            
            # 显示所有可用视频流
            for stream_type, frame in frames.items():
                if frame is not None:
                    cv2.imshow(stream_type.capitalize() + ' View', frame)
            
            # 统一刷新显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            
            time.sleep(0.01)  # 控制刷新率
        cv2.destroyAllWindows()
    
    def show_performance_overlay(self, frame):
        fps = self.frame_counter / (time.time() - self.start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Buffer: {self.raw_queue.qsize()}/{self.processed_queue.qsize()}",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    def _draw_tennis_markers(self, frame, coordinates):
        """绘制网球轨迹（示例实现）"""
        for coord in coordinates:
            x, y = int(coord[0]), int(coord[1])
            cv2.circle(frame, (x,y), 10, (0,0,255), -1)
        return frame        
    
    def _draw_player_boxes(self, frame, results):
        """绘制球员框"""
        _, x1, y1, x2, y2, x3, y3, x4, y4 = results
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.rectangle(frame, (x3,y3), (x4,y4), (0,255,0), 2)
        return frame
    
    def run(self):
        # 启动视频采集
        self.stream_active.set()
        capture_thread = threading.Thread(target=self.video_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()

        # 等待首帧
        while self.raw_queue.empty():
            time.sleep(0.1)
        print("First frame received")

        # 启动处理流程
        self.processing_started.set()
        preprocessing_thread = threading.Thread(target=self.preprocessing_thread)
        tennis_thread = threading.Thread(target=self.tennis_detection_thread)
        player_thread = threading.Thread(target=self.player_detection_thread)
        viz_thread = threading.Thread(target=self.visualization_thread)

        for t in [preprocessing_thread, tennis_thread, player_thread, viz_thread]:
            t.daemon = True
            t.start()

        # 主线程监控
        try:
            while True:
                time.sleep(1)
                status = f"Raw[{self.raw_queue.qsize()}] Processed[{self.processed_queue.qsize()}]"
                status += f" Viz[{sum(q.qsize() for q in self.viz_queues.values())}]"
                print(status)
        except KeyboardInterrupt:
            self.shutdown()
            
    def shutdown(self):
        """安全关闭"""
        print("Shutting down...")
        self.stream_active.clear()
        self.processing_started.clear()
        cv2.destroyAllWindows()
            
if __name__ == "__main__":
    pipeline = Pipeline(max_queue_size=15)
    pipeline.run()

# def main():
#     video_stream = VideoStreamer()
#     # video_stream = cv2.VideoCapture('/home/max/Desktop/tennis/tennis-v5/data/InputVideos/input.mp4')
#     frame_processor = FrameProcessor()
#     tennis_tracker = TennisTracker()
#     # tennis_tracker = TennisTracker_time_delay_profiling()
#     player_tracker = PlayerTracker()
#     frames_delay_buffer = []
#     process_delay_buffer = []
#     i = 0
#     start_time = time.time()
#     while i < 1000:
#     # while True:
#         frame = video_stream.get_frame()
#         # video_stream.frame_number += 1
#         # frame_for_processing = frame.copy()
#         # ret, frame = video_stream.read()
#         original_frame, frame_for_processing = frame_processor.process(frame)
#         # cv2.imshow('frame', frame_for_processing)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
    
#         frames_delay_buffer.append(original_frame)
#         process_delay_buffer.append(frame_for_processing)
        
#         if len(frames_delay_buffer) == 3:
#             player_result = player_tracker.process_frame(process_delay_buffer)
#             coordinates = tennis_tracker.process(process_delay_buffer)
#             for result in player_result:
#                 print(result)

#             for coordinate in coordinates:
#                 print(coordinate)
#             # print(coordinates)
#             # inncer_counter = 0
#             # for coordinate in coordinates:
#                 # print(coordinate)
#                 # x = coordinate[3]
#                 # y = coordinate[4]
#         #         # cv2.circle(frames_delay_buffer[inncer_counter], (int(x), int(y)), 3, (0, 0, 255), 2)
#         # #         # cv2.imshow('frame', frames_delay_buffer[inncer_counter])
#                 # inncer_counter += 1
#             frames_delay_buffer = []
#             process_delay_buffer = []
        
#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     break
#         i += 1
#     # tennis_tracker.print_timing_summary()
#     end_time = time.time()
#     print(f"Time taken average: {(end_time - start_time)} ms")
#     # cv2.destroyAllWindows()
#     #     for t in player_tracker.track_history.values():
#     #         if len(t['positions']) > 0:
#     #             x1, y1, x2, y2 = t['positions'][-1].astype(int)
#     #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     #     # cv2.imshow('masked_frame', masked_frame)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break
#     # cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
