import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from tqdm import tqdm
from scipy.signal import butter, filtfilt

def normalize(signal):
    max_val = np.max(np.abs(signal)) # Ermittelt den maximalen Wert 
    x = signal / max_val if max_val != 0 else signal # Dividiert die Werte durch maximalen Wert
    return x

def segment_cross_correlation(data, sample_rate, sync_length):
    segment = data[:sync_length] # Bereich in dem Sync Ton ist
    correlations = correlate(normalize(segment[:, 0]), normalize(segment[:, 1]), mode='full') # führt correlation in lib aus
    return correlations

def find_offset(correlation):
    offset = correlation.argmax() - (len(correlation) // 2) # berechnet verschobene Samples
    return offset

def modify_data(data, offset):
    if offset > 0:
        new_data = np.column_stack((data[abs(offset):, 0], data[:-abs(offset), 1])) # schneidet von beiden Audiospuren den offset weg 
    elif offset < 0:
        new_data = np.column_stack((data[:-abs(offset), 0], data[abs(offset):, 1])) # schneidet von beiden Audiospuren den offset weg 
    else:
        new_data = data
    return new_data

def high_pass_filter(data, sample_rate, cutoff=20000):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    # Order of the filter
    order = 5
    # Design a high-pass filter using the butterworth design
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def process_wav_file(wav_filename, sync_duration_ms):
    sample_rate, data = wavfile.read(wav_filename) # ließt wav file aus (Messungen pro sec) (Datenpunkte als Array)
    sync_length = int(sample_rate * sync_duration_ms / 1000) # berechnet länge des sync tons
    if data.shape[1] != 2: # wenn 2 Channels nicht vorhanden
        raise ValueError(f"WAV file {wav_filename} is not stereo")
    correlations = segment_cross_correlation(data, sample_rate, sync_length) # führt correlation aus
    offset = find_offset(correlations) # berechnet offset

    data = data[sync_length:] # schneidet sync-ton weg
    new_data = modify_data(data, offset) # synchronisiert die einzelnen Tonspuren mittels offset

    # Apply high-pass filter to remove frequencies below 20kHz
    filtered_data = high_pass_filter(new_data, sample_rate, cutoff=20000)

    print(f"File: {file}, Offset: {offset}")

    return filtered_data, sample_rate 

if len(sys.argv) > 1:
    folder_path = sys.argv[1]  # First command-line argument
    if os.path.exists(folder_path):
        print(f"The folder '{folder_path}' exists.")
        # Count the number of .wav files
        wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
        total_files = len(wav_files)
        print(f"Processing {total_files} files...")
    else:
        print(f"The folder '{folder_path}' does not exist.")
        sys.exit(1)  # Exit if folder does not exist
else:
    print("Please provide a folder path as a command-line argument.")
    sys.exit(1)  # Exit if no argument is provided

new_directory = os.path.join(folder_path, "processed")
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Use tqdm for the progress bar, iterating over wav_files
for file in tqdm(wav_files, desc="Processing", unit="file"):
    base_filename = file.split('.', 1)[0]
    processed_file_path = os.path.join(new_directory, base_filename + ".wav")

    # Skip if the file already exists
    if os.path.exists(processed_file_path):
        continue

    wav_file_path = os.path.join(folder_path, file)
    try:
        new_data, sample_rate = process_wav_file(wav_file_path, 100)
        wavfile.write(processed_file_path, sample_rate, new_data.astype(np.int16))
    except Exception as e:
        print(f"Error processing file {file}: {e}")
