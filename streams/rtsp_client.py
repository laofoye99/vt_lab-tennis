import cv2
import json
import threading
import requests
import time

class RTSPVideoStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        if not self.capture.isOpened():
            raise Exception("无法打开RTSP流")
    
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        time.sleep(1/self.fps)
        return self
    
    def update(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                print("RTSP流断开，尝试重连中...")
                time.sleep(1)
                self.capture = cv2.VideoCapture(self.rtsp_url)
                continue

            with self.lock:
                self.frame = frame
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
        
    def stop(self):
        self.stopped = True
        self.capture.release()

class VideoStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = UrlProcessor().rtsp_url
        self.capture = cv2.VideoCapture(self.rtsp_url)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.stop_event = threading.Event()
        self.current_frame = None
        self.lock = threading.Lock()

    def start_stream(self):
        def _thread_func():
            while not self.stop_event.is_set():
                ret, frame = self.capture.read()
                if ret:
                    with self.lock:
                        self.current_frame = frame
                time.sleep(1/self.fps)
        
        self.thread = threading.Thread(target=_thread_func)
        self.thread.start()

class UrlProcessor:
    def __init__(self):
        self.base_url = "http://jason-shawjaine.top:7001/tennisVideo"
        self.rtsp_url = self.get_rtsp_url()

    def get_rtsp_url(self):
        """获取RTSP_URL"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == 200:
                inner_data = data.get('data', {})
                self.rtsp_url = inner_data.get('videoUrl')
                
                if self.rtsp_url:
                    print(f"成功获取RTSP_URL")
                    return self.rtsp_url
                else:
                    print(f"错误: 无法从API响应中获取RTSP_URL")
                    return None
            else:
                print(f"错误: API响应码为{data.get('code')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"错误：请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败: {e}")
            return None
        except Exception as e:
            print(f"错误: 未知错误: {e}")
            return None

url_processor = UrlProcessor()
rtsp_url = url_processor.rtsp_url
stream = RTSPVideoStream(rtsp_url).start()
while True:
    frame = stream.read()
    if frame is not None:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.stop()
                break

cv2.destroyAllWindows()