# /utils/time_utils.py

import time
import numpy as np
from datetime import datetime, timezone, timedelta

def get_beijing_time():
    utc_time = datetime.now(timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    beijing_time = utc_time.astimezone(beijing_tz)
    return beijing_time

def frameNum2timeStamp(frame_num, fps=25):
    """将帧号转换为时间戳"""
    
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("fps 必须是大于0的整数。")
    
    # 计算时间间隔
    time_delta = int(1000 / fps)
    
    if isinstance(frame_num, np.ndarray):
        timestamp = (frame_num * time_delta).astype(np.int64)
        return timestamp
    elif isinstance(frame_num, int):
        timestamp = (frame_num * time_delta)
        return timestamp
    else:
        raise TypeError("frame_num 必须是整数或 NumPy 数组。")
    
def createTimeStamp():
    timestamp = int(time.perf_counter() * 1000)
    return timestamp
