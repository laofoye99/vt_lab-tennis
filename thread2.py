import cv2
import threading
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.tennis_tracker.inference import TennisTracker
from models.player_tracker.player_tracker import PersonDetector
class VideoProcessingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.video_stream = VideoStream()
        self.frame_processor = FrameProcessor()
        self.tennis_tracker = TennisTracker()
        self.person_detector = PersonDetector()
        
        self.frame = None
        self.processed_frame = None
        self.tennis_tracking_result = None
        self.person_tracking_result = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            frame = self.video_stream.read()
            if frame is None:
                break
            
            # 预处理帧
            processed_frame = self.frame_processor.process(frame)
            
            # 追踪网球轨迹
            tennis_tracking_result = self.tennis_tracker.track(processed_frame)
            
            # 追踪人物轨迹
            person_tracking_result = self.video_stream.track(frame)
            
            # 存储结果
            self.frame = frame
            self.processed_frame = processed_frame
            self.tennis_tracking_result = tennis_tracking_result
            self.person_tracking_result = person_tracking_result

    def stop(self):
        self.stopped = True

class DisplayThread(threading.Thread):
    def __init__(self, processing_thread):
        super().__init__()
        self.processing_thread = processing_thread
        self.stopped = False

    def run(self):
        while not self.stopped:
            frame = self.processing_thread.frame
            processed_frame = self.processing_thread.processed_frame
            tennis_tracking_result = self.processing_thread.tennis_tracking_result
            person_tracking_result = self.processing_thread.person_tracking_result
            
            if frame is not None and processed_frame is not None:
                # 显示原始帧
                cv2.imshow("Original Frame", frame)
                
                # 显示预处理后的帧
                cv2.imshow("Processed Frame", processed_frame)
                
                # 显示网球追踪结果
                if tennis_tracking_result is not None:
                    for bbox in tennis_tracking_result:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Tennis Tracking", frame)
                
                # 显示人物追踪结果
                if person_tracking_result is not None:
                    for track in person_tracking_result:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        bbox = list(map(int, ltrb))
                        cv2.rectangle(person_tracking_result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(person_tracking_result, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow("Person Tracking", person_tracking_result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

    def stop(self):
        self.stopped = True

def main():
    # 打开摄像头或视频文件
    vs = VideoStream().start()  # 使用摄像头，如果要使用视频文件，可以替换为 'path_to_video.mp4'
    
    # 初始化处理器和追踪器
    fp = FrameProcessor()
    tt = TennisTracker()
    ps = PersonDetector()
    
    # 创建并启动视频处理线程
    processing_thread = VideoProcessingThread()
    processing_thread.start()
    
    # 创建并启动显示线程
    display_thread = DisplayThread(processing_thread)
    display_thread.start()
    
    try:
        while not display_thread.stopped:
            pass
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        display_thread.stop()
        processing_thread.stop()
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
