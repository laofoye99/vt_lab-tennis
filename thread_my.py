import threading
import time
import cv2
from multiprocessing import Process, set_start_method, Queue, shared_memory, Lock
import torch
import numpy as np
from collections import deque
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.player_tracker.player_tracker import PersonDetector
from models.tennis_tracker.inference import TennisTracker
from utils.shared_data import SharedData

class VideoCaptureThread(SharedData):
    def __init__(self, max_queue_size=30):
        super().__init__()
        self.video_stream = VideoStream()
        self.running = False
        self.lock = threading.Lock()
        self.frame_queue = deque(maxlen=max_queue_size)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def update(self):
        while self.running:
            frame = self.video_stream.get_frame()
            if frame is not None:
                with self.lock:
                    self.frame_queue.append(frame)
            time.sleep(0.01)

    def get_latest_frame(self):
        with self.lock:
            return self.frame_queue[-1].copy() if self.frame_queue else np.zeros((1280, 720, 3), dtype=np.uint8)

    def get_recent_frames(self, num_frames):
        with self.lock:
            return list(self.frame_queue)[-num_frames:] if len(self.frame_queue) >= num_frames else list(self.frame_queue)

class FrameProcessorThread(SharedData):
    def __init__(self, video_capture_thread, max_queue_size=30):
        super().__init__()
        self.video_capture_thread = video_capture_thread
        self.frame_processor = FrameProcessor()
        self.processing_queue = deque(maxlen=max_queue_size)
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        
    def update(self):
        while self.running:
            frame = self.video_capture_thread.get_latest_frame()
            if frame is not None:
                processed = self.frame_processor.process(frame)
                with self.lock:
                    self.processing_queue.append(processed)
            time.sleep(0.04)
            
    def get_latest_frame(self):
        with self.lock:
            return self.processing_queue[-1].copy() if self.processing_queue else np.zeros((1280, 720, 3), dtype=np.uint8)
            
    def get_recent_frames(self, num_frames):
        with self.lock:
            return list(self.processing_queue)[-num_frames:] if len(self.processing_queue) >= num_frames else list(self.processing_queue)

class SharedFrameData:
    def __init__(self):
        self.shm_original = shared_memory.SharedMemory(create=True, size=1280 * 720 * 3)
        self.original_frame = np.ndarray((1280, 720, 3), dtype=np.uint8, buffer=self.shm_original.buf)
        
        self.shm_processed = shared_memory.SharedMemory(create=True, size=1280 * 720 * 3)
        self.processed_frame = np.ndarray((1280, 720, 3), dtype=np.uint8, buffer=self.shm_processed.buf)
        
        self.lock = Lock()

def process_frame_task(shared_data):
    processor = FrameProcessor()
    while True:
        with shared_data.lock:
            original_frame = shared_data.original_frame.copy()
        processed_frame = processor.process(original_frame)
        with shared_data.lock:
            np.copyto(shared_data.processed_frame, processed_frame)
            
      
def run_tennis_tracker(shared_data):
    tennis_tracker = TennisTracker()
    try:
        while True:
            with shared_data.lock:
                original_frame = shared_data.original_frame.copy()
                processed_frame = shared_data.processed_frame.copy()
            tennis_tracker.process(processed_frame, original_frame, event_draw=True)
    except KeyboardInterrupt:
        pass
    finally:
        tennis_tracker.cleanup()
        
def run_player_tracker(shared_data):
    player_tracker = PersonDetector()
    try:
        while True:
            with shared_data.lock:
                original_frame = shared_data.original_frame.copy()
                processed_frame = shared_data.processed_frame.copy()
            player_tracker.detect(processed_frame, original_frame, event_draw=True)
    except KeyboardInterrupt:
        pass
    finally:
        player_tracker.cleanup()

def capture_task(shared_data):
    video_capture_thread = VideoCaptureThread()
    video_capture_thread.start()
    while True:
        frame = video_capture_thread.get_latest_frame()
        with shared_data.lock:
            np.copyto(shared_data.original_frame, frame)

def process_frame_task(raw_frame_queue, processed_frame_queue):
    processor = FrameProcessor()
    while True:
        if not raw_frame_queue.empty():
            frame = raw_frame_queue.get()
            processed = processor.process(frame)
            processed_frame_queue.put(processed)

def main():
    set_start_method('spawn')
    
    shared_data = SharedFrameData()
    
    # raw_frame_queue = Queue(maxsize=30)
    # processed_frame_queue = Queue(maxsize=30)
    
    processes = [
        Process(target=capture_task, args=(shared_data,)),
        Process(target=process_frame_task, args=(shared_data,)),
        Process(target=run_tennis_tracker, args=(shared_data,)),
        Process(target=run_player_tracker, args=(shared_data,)),
    ]
    
    for process in processes:
        process.start()
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
    finally:
        shared_data.shm_original.close()
        shared_data.shm_processed.close()
        shared_data.shm_original.unlink()
        shared_data.shm_processed.unlink()
    
if __name__ == "__main__":
    main()
