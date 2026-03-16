import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from utils.trajectory import Trajectory, EnhancedTrajectory, fill_continuous_trajectory
import math

class RealTimeSegmenter:
    def __init__(self, crossing_callback, m_line=344):  # 1280: 344, 2560: 670
        self.m_line = m_line  # 分割线
        self.crossing_callback = crossing_callback
        self.in_segment = False
        self.current_segment = None  # 'low' 或 'high'
        self.buffer = []
        self.extrema_ratio_threshold = 0.25

    def process_row(self, frame_number, x, y):
        # 记录所有输入，包括 NaN
        self.buffer.append([frame_number, x, y])

        # 检查是否为 NaN
        if math.isnan(y):
            return None

        # 如果当前不在任何分段中，判断是否开始新分段
        if not self.in_segment:
            if y < self.m_line:
                self.current_segment = 'low'
                self.in_segment = True
            elif y > self.m_line:
                self.current_segment = 'high'
                self.in_segment = True
            # 开始分段时，保留当前行在 buffer 中
            return None

        # 如果正在分段中，判断是否结束
        if self.current_segment == 'low' and y > self.m_line:
            # 低段结束，调用回调
            return self._end_segment()
        elif self.current_segment == 'high' and y < self.m_line:
            # 高段结束，调用回调
            return self._end_segment()

        # 未结束，继续收集
        return None

    def _end_segment(self):
        # 将 buffer 转为 DataFrame 并调用回调
        df = pd.DataFrame(self.buffer, columns=['frame_number', 'x', 'y'])
        results = None
        # print(df)

        if self.crossing_callback:
            # 回调函数接收 DataFrame 和 segment 类型
            results = self.crossing_callback(df, self.current_segment)

        # 重置状态
        self.in_segment = False
        self.current_segment = None
        self.buffer = []

        return results
    

def detect_high_frequency_fluctuations(y, extrema_ratio_threshold):
    """
    检测序列是否具有高频率波动。

    Args:
        y: 信号的 Y 值数据，一维 NumPy 数组。
        extrema_ratio_threshold: 局部极值点数量与标准差之比的阈值。
                                 高于此阈值认为存在高频波动。

    Returns:
        is_high_frequency: 布尔值，表示是否检测到高频率波动。
        extrema_count: 局部极值点总数。
        std_dev: 数据 Standard Deviation.
        extrema_std_ratio: 局部极值点数量与标准差之比。
    """
    if len(y) < 10:
        # print(f'less 10 !!!!') # Debug info
        return False, 0, 0, 0

    std_dev = np.std(y)
    if std_dev < 1e-6:
        return False, 0, std_dev, 0

    peaks, _ = find_peaks(y)
    troughs, _ = find_peaks(-y)
    extrema_count = len(peaks) + len(troughs)
    extrema_std_ratio = extrema_count / std_dev
    is_high_frequency = False
    if extrema_std_ratio > extrema_ratio_threshold:
        is_high_frequency = True
    return is_high_frequency, extrema_count, std_dev, extrema_std_ratio

def find_variance_drop_point_in_segment(y_data_segment, frame_numbers_segment, window_size, min_variance_drop_threshold):
    """
    Finds the significant variance drop point within a segment.

    Args:
        y_data_segment: Segment's Y values.
        frame_numbers_segment: Segment's frame numbers (can be pandas Series or NumPy array).
        window_size: Sliding window size.
        min_variance_drop_threshold: Threshold for significant variance drop (negative).

    Returns:
        split_frame_number: The frame number to split at, or None if no significant drop is found.
        steepest_drop_value: The value of the steepest variance drop.
    """
    n_frames = len(y_data_segment)
    if n_frames < window_size:
        # print(f"Segment too short ({n_frames}) for window size {window_size}. Cannot find variance drop.") # Debug info
        return None, 0 # Cannot apply sliding window if segment is too short

    variances = []
    feature_frame_numbers = [] # Frame numbers corresponding to window centers

    # Slide the window over the segment
    for i in range(n_frames - window_size + 1):
        window_data = y_data_segment[i : i + window_size]
        # Calculate center index in the original segment's data (positional index)
        center_index_in_segment = i + window_size // 2
        # Get the corresponding frame number using positional indexing (.iloc)
        # Ensure index is within bounds of the segment's frame numbers
        if center_index_in_segment < len(frame_numbers_segment):
             # **已修复：使用 .iloc[] 进行位置索引**
             if isinstance(frame_numbers_segment, pd.Series):
                 feature_frame_numbers.append(frame_numbers_segment.iloc[center_index_in_segment])
             else: # Assume it's a NumPy array or list
                 feature_frame_numbers.append(frame_numbers_segment[center_index_in_segment])
        # Calculate variance for the window
        variances.append(np.var(window_data))

    split_frame_number = None
    steepest_drop_value = 0

    # Ensure we have enough variance values to calculate difference
    if len(variances) > 1:
        variance_diff = np.diff(variances)

        # Find the index of the steepest variance drop
        min_diff_index = np.argmin(variance_diff)
        steepest_drop_value = variance_diff[min_diff_index]

        # Check if the steepest drop is below the significance threshold
        if min_variance_drop_threshold is not None and steepest_drop_value < min_variance_drop_threshold:
             # The split point is estimated at the frame number corresponding to the center of the window
             # *after* the steepest drop occurred. This is associated with feature_frame_numbers[min_diff_index].
             # Ensure the index is valid within the feature_frame_numbers
             if min_diff_index < len(feature_frame_numbers):
                  # The find_variance_drop_point_in_segment function should return just the frame number
                  # Let's correct its return value to be just the frame number
                  split_frame_number = feature_frame_numbers[min_diff_index]
             # else:
                 # print(f"Debug: min_diff_index {min_diff_index} out of bounds for feature_frame_numbers length {len(feature_frame_numbers)}")

    # Corrected return: only return the split frame number and steepest drop value separately
    return split_frame_number, steepest_drop_value

# Main segmentation function (used within shot_and_bounce)
def segment_if_large_amplitude_and_high_frequency(y_data, frame_numbers, range_threshold, extrema_ratio_threshold, window_size, min_variance_drop_threshold):
    """
    Segments a sequence if it has a large amplitude range and detects high-frequency fluctuations,
    splitting at the point of the most significant variance drop.

    Args:
        y_data: The segment's Y values (NumPy array or pandas Series).
        frame_numbers: The segment's frame numbers (NumPy array or pandas Series).
        range_threshold: Minimum max-min range to consider for segmentation.
        extrema_ratio_threshold: Threshold for detecting high-frequency fluctuations (for detect_high_frequency_fluctuations).
        window_size: Sliding window size for variance drop analysis within the segment.
        min_variance_drop_threshold: Threshold for significant variance drop (negative) within the segment.

    Returns:
        A tuple:
            - A list of segments. Each segment is a list of frame numbers.
              Returns [original_segment_frame_numbers] if no segmentation occurs.
              Returns [segment1_frame_numbers, segment2_frame_numbers] if segmented.
            - The index (0-based) of the segment considered the 'large amplitude' part.
              This will be 0 if the initial conditions for segmentation were met.
    """
    y_data_arr = np.asarray(y_data)
    frame_numbers_arr = np.asarray(frame_numbers)

    if len(y_data_arr) != len(frame_numbers_arr):
        print("Error: y_data and frame_numbers must have the same length.")
        return [list(frame_numbers_arr)], 0

    if len(y_data_arr) == 0:
        return [], -1

    # The amplitude_range check is still relevant to see if a "large amplitude" part exists in the first place.
    amplitude_range = np.max(y_data_arr) - np.min(y_data_arr)

    # **Added Debug Print for Amplitude Range**
    # print(f"--- Debugging segment_if_large_amplitude_and_high_frequency ---")
    # print(f"Segment frames: {frame_numbers_arr[0]} to {frame_numbers_arr[-1]}")
    # print(f"Amplitude Range: {amplitude_range:.2f}, Threshold: {range_threshold}")


    if amplitude_range < range_threshold:
        # If the overall segment doesn't have large amplitude, no segmentation is needed based on this criteria
        print(f"Amplitude range below threshold. No segmentation.")
        print(f"-------------------------------------------------------")
        return [list(frame_numbers_arr)], 0

    # **Removed Redundant High Frequency Check Here**
    # The decision to call this function is already based on the high frequency check
    # performed earlier in shot_and_bounce on the initial_segment_data.


    # If amplitude condition is met, try to find a split point based on variance drop
    split_frame_number, steepest_drop_value_found = find_variance_drop_point_in_segment(y_data_arr, frame_numbers_arr, window_size, min_variance_drop_threshold)

    # **Added Debug Print for Variance Drop Search Result**
    print(f"Variance drop search result: split_frame_number={split_frame_number}, steepest_drop_value={steepest_drop_value_found:.2f}, Threshold: {min_variance_drop_threshold}")


    if split_frame_number is not None:
        # Check if split_frame_number is within the range of frame_numbers in this segment
        if frame_numbers_arr[0] <= split_frame_number <= frame_numbers_arr[-1]:
            split_index_in_segment = np.where(frame_numbers_arr >= split_frame_number)[0]
            if len(split_index_in_segment) > 0:
                 split_index_in_segment = split_index_in_segment[0]
            else:
                 split_index_in_segment = len(frame_numbers_arr)

            segment1_frames = frame_numbers_arr[:split_index_in_segment]
            segment2_frames = frame_numbers_arr[split_index_in_segment:]

            if len(segment1_frames) > 0 and len(segment2_frames) > 0:
                 # If segmented, the first part is the large amplitude one by definition of the split point
                 print(f"Split point found at frame {split_frame_number}. Segmenting.")
                 print(f"-------------------------------------------------------")
                 return [list(segment1_frames), list(segment2_frames)], 0
        # else:
            # print(f"Calculated split frame {split_frame_number} is outside the valid range for splitting within segment frames {frame_numbers_arr[0]}-{frame_numbers_arr[-1]}. No segmentation.")

    # If no significant variance drop found or splitting resulted in invalid segments
    # Return the original segment as the "large amplitude" part if the initial amplitude check passed but no split point was found
    print(f"No valid split point found. No segmentation.")
    print(f"-------------------------------------------------------")
    return [list(frame_numbers_arr)], 0


# --- Your original shot_and_bounce function with integrated segmentation ---

def shot_and_bounce(df, type_s,
                    range_threshold=150, # Default segmentation parameters for segment_if_large_amplitude_and_high_frequency
                    extrema_ratio_threshold=0.28, # Threshold for detect_high_frequency_fluctuations (using user's value)
                    window_size_variance=20, # Default segmentation parameters for segment_if_large_amplitude_and_high_frequency
                    min_variance_drop_threshold_segment=-10): # Default segmentation parameters for segment_if_large_amplitude_and_high_frequency
    """
    Processes trajectory data to find shot and bounce points,
    with optional segmentation based on amplitude and frequency.

    Args:
        df: Input DataFrame with trajectory data (including 'frame_number', 'center_x', 'center_y').
        type_s: Type of shot ('low' or 'high').
        range_threshold: Minimum max-min range for segmentation (used in segment_if_large_amplitude_and_high_frequency).
        extrema_ratio_threshold: Threshold for high-frequency detection (used in detect_high_frequency_fluctuations).
        window_size_variance: Window size for variance drop analysis (used in segment_if_large_amplitude_and_high_frequency).
        min_variance_drop_threshold_segment: Threshold for significant variance drop (used in segment_if_large_amplitude_and_high_frequency).

    Returns:
        A tuple:
            - bounce: A list of bounce dictionaries, or empty list.
            - shot: Dictionary with shot information, or empty dict.
            - stop_event_detected: Boolean, True if a stop event (segmentation) was detected, False otherwise.
    """
    bounces = [] # Changed to a list to hold multiple bounces
    shot = {}
    b = None
    s = None # Initialize s
    stop_event_detected = False # Flag for stop event

    traj = Trajectory(df)
    turning_points = traj.detect_turning_points(
        angle_threshold=30,
        filter_parabola=True
    )
    enhance_traj = EnhancedTrajectory(df, turning_points)
    enhance_traj.adaptive_interpolate()
    en_df = enhance_traj.get_full_df()
    detect = trajectory2()

    segments = detect.run(en_df) # Initial segmentation by trajectory2

    # --- Get the relevant initial segment data (DataFrame or Series) ---
    initial_segment_data = None

    if type_s == 'low':
        if len(segments.get('low_segments', [])) > 0:
            initial_segment_data = segments['low_segments'][0]
    else: # type_s == 'high'
        if len(segments.get('high_segments', [])) > 0:
            initial_segment_data = segments['high_segments'][0]

    # Ensure we have initial segment data and create a copy
    if initial_segment_data is None:
        print(f"No initial segment data found for type '{type_s}'.")
        return bounces, shot, stop_event_detected # Return empty bounces list

    # **Modified: Directly copy initial_segment_data**
    initial_segment_df = initial_segment_data.copy()


    if initial_segment_df.empty:
        print(f"Initial segment DataFrame is empty for type '{type_s}'.")
        return bounces, shot, stop_event_detected # Return empty bounces list

    # --- Apply fill_continuous_trajectory first ---
    recovery_df = fill_continuous_trajectory(initial_segment_df)

    if recovery_df.empty:
         print(f"Recovery DataFrame is empty after fill_continuous_trajectory for type '{type_s}'.")
         return bounces, shot, stop_event_detected # Return empty bounces list

    # --- Check for high frequency on the *original* initial segment data ---
    # Using initial_segment_data['center_y'] as per user's observation
    is_high_frequency, extrema_count, std_dev, extrema_std_ratio = detect_high_frequency_fluctuations(
        initial_segment_data['center_y'], # Use the original segment data
        extrema_ratio_threshold=extrema_ratio_threshold # Use the passed threshold
    )

    # --- Determine the segment DataFrame(s) to process for shot/bounce detection ---
    segment_df_for_detection = pd.DataFrame(columns=recovery_df.columns) # Initialize empty
    smaller_amplitude_segment_df = pd.DataFrame(columns=recovery_df.columns) # Initialize empty
    perform_shot_detection = True # Flag to control if shot detection is performed


    if is_high_frequency:
        print(f"High frequency detected on initial segment data (Ratio: {extrema_std_ratio:.2f}). Applying secondary segmentation on recovery_df.")
        # If high frequency is detected, then apply the secondary segmentation logic on recovery_df
        # Pass the recovery_df's center_y and frame_number to the segmentation function
        segmented_results_frames, large_amplitude_segment_index = segment_if_large_amplitude_and_high_frequency(
            recovery_df['center_y'],
            recovery_df['frame_number'],
            range_threshold,
            extrema_ratio_threshold, # Pass the threshold (though it's redundant inside the function now)
            window_size_variance,
            min_variance_drop_threshold_segment
        )

        if len(segmented_results_frames) > 1:
            # Secondary segmentation occurred (split into two parts)
            print(f"Segmentation performed. {len(segmented_results_frames)} parts found:")
            for i, segment_frames in enumerate(segmented_results_frames):
                if len(segment_frames) > 0:
                    print(f"  Segment {i+1}: Frames {segment_frames[0]} to {segment_frames[-1]}")

            stop_event_detected = True # Signal stop event for the smaller amplitude part
            perform_shot_detection = True # Shot detection is performed on the large amplitude segment

            # Get the DataFrame for the large amplitude segment (for shot/bounce detection)
            large_amplitude_segment_frames = segmented_results_frames[large_amplitude_segment_index]
            if large_amplitude_segment_frames:
                segment_df_for_detection = recovery_df[recovery_df['frame_number'].isin(large_amplitude_segment_frames)].copy()

            # Get the DataFrame for the smaller amplitude segment (for bounce detection)
            # Assuming the other segment is the smaller amplitude one
            smaller_amplitude_segment_frames = segmented_results_frames[1 - large_amplitude_segment_index] # Get the index of the other segment
            if smaller_amplitude_segment_frames:
                 smaller_amplitude_segment_df = recovery_df[recovery_df['frame_number'].isin(smaller_amplitude_segment_frames)].copy()

            print("Secondary segmentation detected a stop event.")

        else:
            # High frequency detected, but secondary segmentation did not split.
            # Use the entire recovery_df for detection, but skip shot detection.
            print("High frequency detected, but secondary segmentation did not split. Skipping shot detection.")
            segment_df_for_detection = recovery_df.copy() # Use the whole recovery_df
            stop_event_detected = True # No stop event based on secondary segmentation not splitting
            perform_shot_detection = False # **Skip shot detection as requested**

    else:
        # High frequency not detected on initial segment data, use the entire recovery_df for detection.
        # print(f"No high frequency detected on initial segment data (Ratio: {extrema_std_ratio:.2f}). No secondary segmentation.")
        segment_df_for_detection = recovery_df.copy() # Use the whole recovery_df
        stop_event_detected = False # No stop event
        perform_shot_detection = True # Perform shot detection in the normal case

    # --- Perform Shot and Bounce Detection on the determined segment_df_for_detection ---

    if segment_df_for_detection.empty:
         print(f"Segment DataFrame for detection is empty after segmentation logic for type '{type_s}'.")
         return bounces, shot, stop_event_detected # Return empty bounces list

    # Apply shot detection if the flag is True
    if perform_shot_detection:
        if type_s == 'low':
            if (segment_df_for_detection['center_x'] < 0).any() or (segment_df_for_detection['center_y'] < 0).any():
                 s = detect.top_shot(segment_df_for_detection)
            else:
                 s = detect.top_shot(segment_df_for_detection)
            if s:
                shot = s
        else: # type_s == 'high'
            if (segment_df_for_detection['center_x'] < 0).any() or (segment_df_for_detection['center_y'] < 0).any():
                s = detect.bottom_shot(segment_df_for_detection)
                shot = s
            else:
                s = detect.bottom_shot(segment_df_for_detection)
                shot = s # Assign s to shot here too


    # Always perform bounce detection on the main segment_df_for_detection
    if type_s == 'low':
        b = detect.top_wave(segment_df_for_detection)
        if b:

             if isinstance(b, list) and len(b) > 0 and 'peaks' in b[0] and len(b[0]['peaks']) > 0:
                 bounce_index_in_segment_df = b[0]['peaks'][0]
                 if bounce_index_in_segment_df < len(segment_df_for_detection):
                      bounces.append({ # Append to the list
                          'bounce': segment_df_for_detection.iloc[bounce_index_in_segment_df]['frame_number'],
                          'bounce_x': segment_df_for_detection.iloc[bounce_index_in_segment_df]['center_x'],
                          'bounce_y':segment_df_for_detection.iloc[bounce_index_in_segment_df]['center_y'],
                          'bounce_type': 'top'
                      })
                 else:
                      print(f"Bounce index {bounce_index_in_segment_df} out of bounds for segment_df_for_detection DataFrame.")
             else:
                  print("Could not extract valid bounce index from b for type 'low' in main segment.")

    else: # type_s == 'high'
        b = detect.bottom_wave(segment_df_for_detection)
        if b is not None: # Assuming b is an index for 'high' type relative to segment_df_for_detection
            if b < len(segment_df_for_detection):
                bounces.append({ # Append to the list
                     'bounce': segment_df_for_detection.iloc[b]['frame_number'],
                     'bounce_x': segment_df_for_detection.iloc[b]['center_x'],
                     'bounce_y':segment_df_for_detection.iloc[b]['center_y'],
                     'bounce_type': 'bottom'
                 })
            else:
                 print(f"Bounce index {b} out of bounds for segment_df_for_detection DataFrame.")


    # --- If secondary segmentation occurred, perform bounce detection on the smaller amplitude segment ---
    if stop_event_detected and not smaller_amplitude_segment_df.empty:
         print("Detecting bounce in the smaller amplitude segment.")
         # Apply bounce detection specifically to the smaller amplitude segment
         if type_s == 'low': # Assuming 'low' type corresponds to 'top_wave' for bounce
              b_smaller = detect.top_wave(smaller_amplitude_segment_df)
              if b_smaller:
                   if isinstance(b_smaller, list) and len(b_smaller) > 0 and 'peaks' in b_smaller[0] and len(b_smaller[0]['peaks']) > 0:
                        bounce_index_in_smaller_df = b_smaller[0]['peaks'][0]
                        if bounce_index_in_smaller_df < len(smaller_amplitude_segment_df):
                             bounces.append({ # Append to the list
                                 'bounce': smaller_amplitude_segment_df.iloc[bounce_index_in_smaller_df]['frame_number'],
                                 'bounce_x': smaller_amplitude_segment_df.iloc[bounce_index_in_smaller_df]['center_x'],
                                 'bounce_y':smaller_amplitude_segment_df.iloc[bounce_index_in_smaller_df]['center_y'],
                                 'bounce_type': 'top_stop_event' # Indicate it's a bounce from the stop event segment
                             })
                        else:
                             print(f"Bounce index {bounce_index_in_smaller_df} out of bounds for smaller_amplitude_segment_df DataFrame.")
                   else:
                        print("Could not extract valid bounce index from b_smaller for type 'low'.")

         else: # type_s == 'high', assuming 'high' type corresponds to 'bottom_wave' for bounce
              b_smaller = detect.bottom_wave(smaller_amplitude_segment_df)
              if b_smaller is not None: # Assuming b_smaller is an index for 'high' type
                  if b_smaller < len(smaller_amplitude_segment_df):
                       bounces.append({ # Append to the list
                           'bounce': smaller_amplitude_segment_df.iloc[b_smaller]['frame_number'],
                           'bounce_x': smaller_amplitude_segment_df.iloc[b_smaller]['center_x'],
                           'bounce_y':smaller_amplitude_segment_df.iloc[b_smaller]['center_y'],
                           'bounce_type': 'bottom_stop_event' # Indicate it's a bounce from the stop event segment
                       })
                  else:
                       print(f"Bounce index {b_smaller} out of bounds for smaller_amplitude_segment_df DataFrame.")


    # Return bounces (list), shot (dict), and the stop event flag
    return bounces, shot, stop_event_detected


class trajectory2(object):
    """

    """

    def __init__(self, min_order=3, max_order=3):
        """

        """

        # self.x = x
        # self.y = y

        self.t_line = 280  # 1280: 273, 2560: 280
        self.m_line = 344  # 1280: 344, 2560: 670
        self.bottom_line = 1183  # 1280: 593, 2560: 1183

        self.min_order = min_order  # 波谷检测窗口
        self.max_order = max_order  # 波峰检测窗口
    
    def top_shot(self, l_seg:pd.DataFrame):
        """

        """
        bias = 5
        shot = {}
        peaks = find_peaks(-l_seg['center_y'].values)[0]

        if len(peaks)>0:
            shot['shot'] = l_seg.iloc[peaks[-1] - bias]['frame_number']
            shot['shot_x'] = l_seg.iloc[peaks[-1] - bias]['center_x']
            shot['shot_y'] = l_seg.iloc[peaks[-1] - bias]['center_y']
            shot['shot_type'] = 'top'
        return shot

    def top_wave(self, df):
        """
        检测以波谷为顶点的完整波动周期
        返回：波谷间数据段列表和对应波峰数量
        """
        y_values = df.center_y.values
        
        # 检测所有波谷点（局部最小值）
        min_indices = argrelextrema(y_values, np.less, order=self.min_order)[0]
        # print(f"波谷索引: {argrelextrema(y_values, np.less, order=self.min_order)}")
        segments = []
        if len(min_indices) < 2:
            if len(min_indices) == 1:
                # 只有一个波谷
                # print('only single peak')
                min_peaks_indices = find_peaks(-y_values)[0]
                if len(min_peaks_indices) > 1:
                    # print(df.iloc[:min_peaks_indices[0]])
                    
                    if len(min_peaks_indices) < 1:
                        return segments  # 无法形成完整周期
                    s = df.iloc[min_peaks_indices[0]: min_peaks_indices[1]]['center_y'].values
                    max_peaks_indices = np.argmax(s)
                    segments.append({
                        'segment': df.iloc[:max_peaks_indices],
                        'min_start': max_peaks_indices,
                        'min_end': len(df)-1,
                        'peaks': [min_peaks_indices[0] + max_peaks_indices]  # 转换为全局索引
                    })
            return segments  # 不足两个波谷无法形成完整周期
        
        # 遍历相邻波谷对
        for i in range(len(min_indices)-1):
            start_idx = min_indices[i]
            end_idx = min_indices[i+1]
            
            # 提取波谷间数据段
            seg_data = df.iloc[start_idx:end_idx]
            seg_values = seg_data.center_y.values
            
            # 检测波峰（局部最大值）
            max_idx_in_seg = argrelextrema(seg_values, np.greater, 
                                         order=self.max_order)[0]
            
            # 有效性验证（必须存在波峰）
            if len(max_idx_in_seg) > 0:
                segments.append({
                    'segment': seg_data,
                    'min_start': start_idx,
                    'min_end': end_idx,
                    'peaks': max_idx_in_seg + start_idx  # 转换为全局索引
                })
        
        return segments
    
    def bottom_wave(self, h_segment):
        """
        
        """
        turning_index = None
        y = h_segment['center_y'].values
        
        local_min_indices = argrelextrema(y, np.greater)[0]  # 局部最小
        if len(local_min_indices) > 0:
            turning_index = local_min_indices[0]  # 取第一个极小值

        return turning_index
    
    def bottom_shot(self, h_segment):
        """
        
        """
        y = h_segment['center_y'].values
        shot = {}
        bias = 0
        
        peaks = find_peaks(y)[0]
        # print(peaks)
        if len(peaks) > 0:
            shot['shot'] = h_segment.iloc[peaks[-1] - bias]['frame_number']
            shot['shot_x'] = h_segment.iloc[peaks[-1] - bias]['center_x']
            shot['shot_y'] = h_segment.iloc[peaks[-1] - bias]['center_y']
            shot['shot_type'] = 'bottom'
        # print(result)
        return shot
    
    def segment(self, df):
        """分段方法，返回带类型标记的段落"""
        # x = df.center_x
        y = df.center_y
        n = len(y)
        
        # 收集有效分割点及触发类型
        split_info = []
        for i in range(n - 3):
            current = y.iloc[i]
            next_three = y.iloc[i+1:i+4]
            
            if current < self.m_line and next_three.ge(self.m_line).all():
                split_info.append( (i+1, 'low_to_high') )
            elif current > self.m_line and next_three.le(self.m_line).all():
                split_info.append( (i+1, 'high_to_low') )
        
        # 构建完整分割点序列
        split_indices = sorted([s[0] for s in split_info] + [0, n])
        split_indices = list(dict.fromkeys(split_indices))  # 去重保序
        
        # 确定各段落类型
        segments = []
        current_type = 'low' if y.iloc[0] < self.m_line else 'high'
        type_switch = {
            'low_to_high': 'high',
            'high_to_low': 'low'
        }
        
        for i in range(len(split_indices)-1):
            start, end = split_indices[i], split_indices[i+1]
            seg_df = df.iloc[start:end]
            segments.append( (seg_df, current_type) )
            
            # 查找下一个触发类型
            next_triggers = [s[1] for s in split_info if s[0] == end]
            if next_triggers:
                current_type = type_switch[next_triggers[0]]
        
        return segments
        
    def run(self, df):
        """全流程处理入口"""
        segments = self.segment(df)
        results = {'low_segments': [], 'high_segments': []}  # 原top/bottom_segments改为low/high
        for seg_df, seg_type in segments:
            if seg_type == 'low':
                results['low_segments'].append(seg_df)
            else:
                results['high_segments'].append(seg_df)
        
        return results

        