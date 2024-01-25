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

def modify_data(data, offset):
    if offset > 0:
        new_data = np.column_stack((data[abs(offset):, 0], data[:-abs(offset), 1]))
    elif offset < 0:
        new_data = np.column_stack((data[:-abs(offset), 0], data[abs(offset):, 1]))
    else:
        new_data = data
    return new_data


def process_wav_file(wav_filename, sync_duration_ms):
    sample_rate, data = wavfile.read(wav_filename)
    sync_length = int(sample_rate * sync_duration_ms / 1000)
    if data.shape[1] != 2:
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    correlations = segment_cross_correlation(data, sample_rate, sync_length)
    offset = find_offset(correlations)

    data = data[sync_length:]
    new_data = modify_data(data, offset)


    print(f"File: {file}, Offset: {offset}")

    return new_data, sample_rate 

# Folder containing WAV files
folder_path = '/home/imnos/projects/diplomarbeit/software/pi-remote-dir/recordings'
new_directory = os.path.join(folder_path,"processed")

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

while True:
    # Iterate through each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.raw.wav'):
            base_filename = file.split('.', 1)[0]
            processed_file_path = os.path.join(new_directory, base_filename + ".wav")

            # Skip if the file already exists
            if os.path.exists(processed_file_path):
                continue

            wav_file_path = os.path.join(folder_path, file)
            try:
                new_data, sample_rate = process_wav_file(wav_file_path, 15)
                wavfile.write(processed_file_path, sample_rate, new_data.astype(np.int16))

            except Exception as e:
                print(f"Error processing file {file}: {e}")
