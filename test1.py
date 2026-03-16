from multiprocessing import Process, Lock, shared_memory, set_start_method
import torch
import time
import requests
import cv2
import numpy as np
from models.tennis_tracker.inference import TennisTracker
from models.player_tracker.player_tracker import PersonDetector

class SharedFrameData:
    def __init__(self):
        self.shm_original = shared_memory.SharedMemory(create=True, size=1280 * 720 * 3)
        self.original_frame = np.ndarray((1280, 720, 3), dtype=np.uint8, buffer=self.shm_original.buf)
        
        self.shm_processed = shared_memory.SharedMemory(create=True, size=1280 * 720 * 3)
        self.processed_frame = np.ndarray((1280, 720, 3), dtype=np.uint8, buffer=self.shm_processed.buf)
        
        self.lock = Lock()
        
def video_stream(shared_data):
    rtsp_url = "http://jason-shawjaine.top:7001/tennisVideo"
    response = requests.get(rtsp_url)
    data = response.json()
    if data.get('code') == 200:
        inner_data = data.get('data', {})
        rtsp_url = inner_data.get('videoUrl')
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if ret:
            with shared_data.lock:
                np.copyto(shared_data.original_frame, frame)

def process_frame(shared_data):
    black_layer = np.zeros((720, 1280, 3), dtype=np.uint8)
    court_mask = cv2.imread("data/images/court_mask.png", cv2.IMREAD_GRAYSCALE)
    
    while True:
        with shared_data.lock:
            original_frame = shared_data.original_frame.copy()
        darkened_frame = cv2.addWeighted(original_frame, 0, black_layer, 1, 0)
        masked_frame = np.where(court_mask[:, :, np.newaxis] == 255, original_frame, darkened_frame)
        with shared_data.lock:
            np.copyto(shared_data.processed_frame, masked_frame)
            
def run_tennis_tracker(shared_data):
    tennis_tracker = TennisTracker()
    while True:
        with shared_data.lock:
            original_frame = shared_data.original_frame.copy()
            processed_frame = shared_data.processed_frame.copy()
        tennis_tracker.track(processed_frame, original_frame, event_draw=True)

def run_player_tracker(shared_data):
    player_tracker = PersonDetector()
    while True:
        with shared_data.lock:
            original_frame = shared_data.original_frame.copy()
            processed_frame = shared_data.processed_frame.copy()
        player_tracker.detect(processed_frame, original_frame, event_draw=True)
        # result = player_tracker.detect(processed_frame, original_frame, event_draw=True)
        # frame_number, x1, y1, x2, y2, x3, y3, x4, y4 = result
        # cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(original_frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        # cv2.imshow("player frame", original_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def main():
    cv2.setWindowProperty("window_name", cv2.WND_PROP_QT, 1)
    set_start_method("spawn")

    # 初始化共享数据
    shared_data = SharedFrameData()

    processes = [
        Process(target=video_stream, args=(shared_data,)),
        Process(target=process_frame, args=(shared_data,)),
        Process(target=run_tennis_tracker, args=(shared_data,)),
        Process(target=run_player_tracker, args=(shared_data,))
    ]

    for p in processes:
        p.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
    finally:
        shared_data.shm_original.close()
        shared_data.shm_processed.close()
        shared_data.shm_original.unlink()
        shared_data.shm_processed.unlink()
        
if __name__ == "__main__":
    main()
