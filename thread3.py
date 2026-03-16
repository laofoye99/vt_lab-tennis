import threading
import queue
import cv2
import numpy as np
import time
from streams.video_stream import VideoStream
from streams.frame_process import FrameProcessor
from models.tennis_tracker.inference import TennisTracker
from models.player_tracker.player_tracker import PersonDetector
from utils.shared_data import SharedData
from models.speed_calculator.speed_calculator import SpeedCalculator
original_frames = {}
original_frame_lock = threading.Lock()
display_frame_lock = threading.Lock()
display_frame_shared = None
trajectory_image_shared = None
speed_shared = 0
trajectory_points = []
trajectory_points_lock = threading.Lock()

class VideoStreamThread(threading.Thread):
    def __init__(self, raw_queue):
        super().__init__()
        self.raw_queue = raw_queue
        self.video_stream = VideoStream()
        self.running = True
        
    def run(self):
        while self.running:
            frame = self.video_stream.get_frame()
            if frame is not None:
                if self.raw_queue.full():
                    self.raw_queue.get_nowait()
                self.raw_queue.put_nowait(frame)
            else:
                time.sleep(0.001)
        self.video_stream.release()
        
    def stop(self):
        self.running = False
        
class PreprocessThread(threading.Thread, SharedData):
    def __init__(self, raw_queue, processed_queue_person, processed_queue_tennis):
        super().__init__()
        self.raw_queue = raw_queue
        self.processed_queue_person = processed_queue_person
        self.processed_queue_tennis = processed_queue_tennis
        self.frame_processor = FrameProcessor()
        self.running = True

    def run(self):
        while self.running:
            try:
                frame = self.raw_queue.get_nowait()
                processed_frame = self.frame_processor.process(frame)
                current_frame_id = self.frame_number
                self.frame_number += 1
                with original_frame_lock:
                    original_frames[current_frame_id] = frame.copy()   
                    
                if not self.processed_queue_person.full():
                    self.processed_queue_person.put((processed_frame, current_frame_id))
                if not self.processed_queue_tennis.full():
                    self.processed_queue_tennis.put((processed_frame, current_frame_id))    
            except queue.Empty:
                continue

class PersonDetectorThread(threading.Thread):
    def __init__(self, processed_queue, result_queue):
        super().__init__()
        self.processed_queue = processed_queue
        self.result_queue = result_queue
        self.detector = PersonDetector()
        self.running = True

    def run(self):
        while self.running:
            try:
                processed_frame, frame_id = self.processed_queue.get(timeout=1)
                bboxes = self.detector.detect(processed_frame)
                self.result_queue.put((frame_id, bboxes))
            except queue.Empty:
                continue
                
class TennisTrackerThread(threading.Thread):
    def __init__(self, processed_queue, result_queue):
        super().__init__()
        self.processed_queue = processed_queue
        self.result_queue = result_queue
        self.tracker = TennisTracker()
        self.running = True

    def run(self):
        while self.running:
            try:
                processed_frame, frame_id = self.processed_queue.get(timeout=1)
                trajectory = self.tracker.track(processed_frame)
                self.result_queue.put((frame_id, trajectory))
            except queue.Empty:
                continue                
                
class ResultMergerThread(threading.Thread):
    def __init__(self, person_queue, tennis_queue, speed_calculator):
        super().__init__()
        self.person_queue = person_queue
        self.tennis_queue = tennis_queue
        self.speed_calculator = speed_calculator
        self.person_results = {}
        self.tennis_results = {}
        self.running = True

    def run(self):
        while self.running:
            self.process_queue(self.person_queue, self.person_results)
            self.process_queue(self.tennis_queue, self.tennis_results)
            self.try_merge_results()

    def process_queue(self, q, results):
        try:
            frame_id, data = q.get_nowait()
            results[frame_id] = data
        except queue.Empty:
            pass

    def try_merge_results(self):
        common_ids = set(self.person_results.keys()) & set(self.tennis_results.keys())
        for fid in common_ids:
            bboxes = self.person_results.pop(fid)
            trajectory = self.tennis_results.pop(fid)
            with original_frame_lock:
                original_frame = original_frames.pop(fid, None)
            if original_frame is not None:
                display_frame = self.draw_results(original_frame, bboxes, trajectory)
                self.update_trajectory(trajectory)
                speed = self.speed_calculator.calculate(trajectory)
                self.update_display(display_frame, speed)

    def draw_results(self, frame, bboxes, trajectory):
        frame = frame.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for point in trajectory:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        return frame

    def update_trajectory(self, new_points):
        with trajectory_points_lock:
            trajectory_points.extend(new_points)
            h, w = 720, 1280  # 根据实际视频尺寸调整
            if trajectory_image_shared is None:
                self.trajectory_image = np.zeros((h, w, 3), dtype=np.uint8)
            for point in new_points:
                cv2.circle(self.trajectory_image, point, 3, (255, 0, 0), -1)

    def update_display(self, frame, speed):
        with display_frame_lock:
            global display_frame_shared, trajectory_image_shared, speed_shared
            display_frame_shared = frame
            trajectory_image_shared = self.trajectory_image.copy()
            speed_shared = speed        
        
        
def main():
    raw_queue = queue.Queue(maxsize=1)
    processed_person_queue = queue.Queue(maxsize=1)
    processed_tennis_queue = queue.Queue(maxsize=1)
    person_result_queue = queue.Queue()
    tennis_result_queue = queue.Queue()

    speed_calculator = SpeedCalculator()

    video_thread = VideoStreamThread(raw_queue)
    processor_thread = PreprocessThread(raw_queue, processed_person_queue, processed_tennis_queue)
    person_thread = PersonDetectorThread(processed_person_queue, person_result_queue)
    tennis_thread = TennisTrackerThread(processed_tennis_queue, tennis_result_queue)
    merger_thread = ResultMergerThread(person_result_queue, tennis_result_queue, speed_calculator)

    video_thread.start()
    processor_thread.start()
    person_thread.start()
    tennis_thread.start()
    merger_thread.start()

    while True:
        with display_frame_lock:
            display_frame = display_frame_shared.copy() if display_frame_shared is not None else None
            trajectory_img = trajectory_image_shared.copy() if trajectory_image_shared is not None else None
            speed = speed_shared

        if display_frame is not None:
            cv2.putText(display_frame, f"Speed: {speed:.2f} km/h", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Live Tracking", display_frame)
        if trajectory_img is not None:
            cv2.imshow("Ball Trajectory", trajectory_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_thread.stop()
    processor_thread.running = False
    person_thread.running = False
    tennis_thread.running = False
    merger_thread.running = False

    video_thread.join()
    processor_thread.join()
    person_thread.join()
    tennis_thread.join()
    merger_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()