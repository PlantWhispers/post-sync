import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

def normalize(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def segment_cross_correlation(data, sample_rate, sync_length):
    segment = data[:sync_length]
    segment_normalized = normalize(segment)
    correlations = correlate(segment_normalized[:, 0], segment_normalized[:, 1], mode='full')
    return correlations

def find_offset(correlation):
    offset = correlation.argmax() - (len(correlation) // 2)
    return offset

def process_wav_file(wav_filename, sync_duration_ms):
    sample_rate, data = wavfile.read(wav_filename)
    sync_length = int(sample_rate * sync_duration_ms / 1000)
    if data.shape[1] != 2:
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    correlations = segment_cross_correlation(data, sample_rate, sync_length)
    offset = find_offset(correlations)

    data = data[sync_length:]
    if offset > 0:
        new_data = np.column_stack((data[abs(offset):, 0], data[:-abs(offset), 1]))
    elif offset < 0:
        new_data = np.column_stack((data[:-abs(offset), 0], data[abs(offset):, 1]))
    else:
        new_data = data

    base_filename = os.path.basename(wav_filename).split('.', 1)[0]
    new_filename = base_filename + ".wav"
    new_directory = os.path.join(folder_path,"processed")
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    new_path = os.path.join(new_directory, new_filename)
    wavfile.write(new_path, sample_rate, new_data.astype(np.int16))

    return offset 

# Folder containing WAV files
folder_path = '/home/imnos/projects/diplomarbeit/software/pi-remote-dir/recordings'

# Iterate through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.raw.wav'):
        wav_file_path = os.path.join(folder_path, file)
        try:
            start_offset = process_wav_file(wav_file_path, 15)
            print(f"File: {file}, Offset: {start_offset}")
         
        except Exception as e:
            print(f"Error processing file {file}: {e}")
