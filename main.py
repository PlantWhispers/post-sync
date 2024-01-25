import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

def normalize(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def segment_cross_correlation(data, sample_rate, segment_duration_ms):
    segment_length = int(sample_rate * segment_duration_ms / 1000)
    segment = data[:segment_length]
    segment_normalized = normalize(segment)
    correlations = correlate(segment_normalized[:, 0], segment_normalized[:, 1], mode='full')
    return correlations

def find_offset(correlation):
    offset = correlation.argmax() - (len(correlation) // 2)
    return offset

def process_wav_file(wav_filename):
    sample_rate, data = wavfile.read(wav_filename)
    if data.shape[1] != 2:
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    correlations = segment_cross_correlation(data, sample_rate, 15)  # 15 ms
    offset = find_offset(correlations)
    return offset 

# Folder containing WAV files
folder_path = '/home/imnos/projects/diplomarbeit/software/pi-remote-dir/recordings'  # Replace with your folder path

# Iterate through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        wav_file_path = os.path.join(folder_path, file)
        try:
            start_offset = process_wav_file(wav_file_path)
            print(f"File: {file}, Offset: {start_offset}")
         
        except Exception as e:
            print(f"Error processing file {file}: {e}")

