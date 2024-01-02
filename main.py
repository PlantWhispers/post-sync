import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

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

def process_wav_file(wav_filename):
    sample_rate, data = wavfile.read(wav_filename)
    if data.shape[1] != 2:
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    start_corr, end_corr = segment_cross_correlation(data, sample_rate, 200)  # 200 ms
    start_offset = find_offset(start_corr)
    end_offset = find_offset(end_corr)
    return start_offset, end_offset

# Folder containing WAV files
folder_path = 'test_files'  # Replace with your folder path

# Iterate through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        wav_file_path = os.path.join(folder_path, file)
        try:
            start_offset, end_offset = process_wav_file(wav_file_path)
            print(f"File: {file}, Start Offset: {start_offset} samples, End Offset: {end_offset} samples")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
