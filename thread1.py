import cv2
import time
import threading
import queue
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor


class EnhancedPipeline:
    def __init__(self, max_queue_size=25):
        self.video_stream = VideoStream()
        self.frame_processor = FrameProcessor()
        
        self.raw_queue = queue.Queue(maxsize=max_queue_size)
        self.processed_queue = queue.Queue(maxsize=max_queue_size)
        
        self.stream_active = threading.Event()
        
        # self.display_active = threading.Event()
        
        self.display_queues = {
            'original': queue.Queue(maxsize=max_queue_size),
            'processed': queue.Queue(maxsize=max_queue_size)
        }
        

    def video_capture_thread(self):
        """视频流采集线程"""
        print("Video capture thread started")
        while self.stream_active.is_set():
            frame = self.video_stream.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            # original_frame_for_display = frame.copy()
            try:
                self.raw_queue.put_nowait(frame)
                self.display_queues['original'].put_nowait(frame)
            except queue.Full:
                _ = self.raw_queue.get_nowait()
                self.raw_queue.put_nowait(frame)
            
            time.sleep(0.04)
    
    def preprocess_thread(self):
        """预处理线程"""
        print("Preprocessing thread started")
        while self.stream_active.is_set() and not self.raw_queue.empty():
            try:
                frame = self.raw_queue.get(timeout=0.04)
                processed_frame = self.frame_processor.process(frame)
                # print(processed_frame.shape)
                self.processed_queue.put(processed_frame)
                self.display_queues['processed'].put(processed_frame)
            except queue.Empty:
                continue
            except queue.Full:
                _ = self.processed_queue.get_nowait()
                self.processed_queue.put(processed_frame)
                
            time.sleep(0.04)
    
    def display_thread(self, window_name, frame):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown()
            
    def run(self):
        self.stream_active.set()
        capture_thread = threading.Thread(target=self.video_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        while self.raw_queue.empty():
            time.sleep(0.1)
        print("First frame received")
        
        preprocess_thread = threading.Thread(target=self.preprocess_thread)
        preprocess_thread.daemon = True
        preprocess_thread.start()
        
        display_thread = threading.Thread(target=self.display_thread, args=('Original View', self.display_queues['original'].get()))
        display_thread.daemon = True
        display_thread.start()
        
        display_thread = threading.Thread(target=self.display_thread, args=('Processed View', self.display_queues['processed'].get()))
        display_thread.daemon = True
        display_thread.start()
        
        display_thread.join()
        preprocess_thread.join()
        
        try:
            while True:
                time.sleep(1)
                status = f"Viz[{self.raw_queue.qsize()}]"
                print(status)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.stream_active.clear()
        time.sleep(0.5)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = EnhancedPipeline()
    pipeline.run()