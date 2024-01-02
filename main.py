import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

# Constants
SPEED_OF_SOUND = 343  # Speed of sound in air at 20°C in m/s

def normalize(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def segment_cross_correlation(data, sample_rate, segment_duration_ms):
    segment_length = int(sample_rate * segment_duration_ms / 1000)
    start_segment = data[:segment_length]
    end_segment = data[-segment_length:]
    start_segment_normalized = normalize(start_segment)
    end_segment_normalized = normalize(end_segment)
    start_corr = correlate(start_segment_normalized[:, 0], start_segment_normalized[:, 1], mode='full')
    end_corr = correlate(end_segment_normalized[:, 0], end_segment_normalized[:, 1], mode='full')
    return start_corr, end_corr

def find_offset(correlation):
    offset = correlation.argmax() - (len(correlation) // 2)
    return offset

def samples_to_distance(offset, sample_rate):
    time_delay = offset / sample_rate
    distance_meters = SPEED_OF_SOUND * time_delay
    distance_mm = int(distance_meters * 1000)  # Convert to mm and round to integer
    return distance_mm


def process_wav_file(wav_filename):
    sample_rate, data = wavfile.read(wav_filename)
    if data.shape[1] != 2:
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    start_corr, end_corr = segment_cross_correlation(data, sample_rate, 200)  # 200 ms
    start_offset = find_offset(start_corr)
    end_offset = find_offset(end_corr)
    start_distance_mm = samples_to_distance(start_offset, sample_rate)
    end_distance_mm = samples_to_distance(end_offset, sample_rate)
    return start_distance_mm, end_distance_mm

# Folder containing WAV files
folder_path = 'test_files'  # Replace with your folder path

# Iterate through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        wav_file_path = os.path.join(folder_path, file)
        try:
            start_distance, end_distance = process_wav_file(wav_file_path)
            print(f"File: {file}, Start Distance: {abs(start_distance):>3} mm, End Distance: {abs(end_distance):>3} mm")
            # print(f"File: {file}, Start Distance: {start_distance} mm, End Distance: {end_distance} mm")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
