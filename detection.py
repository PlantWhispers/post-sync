import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Constants
Threshold = 0.4

# Load the audio file
if len(sys.argv) > 1:
    folder_path = sys.argv[1]  # First command-line argument
    if os.path.exists(folder_path):
        # Count the number of .wav files
        wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
        total_files = len(wav_files)
    else:
        print(f"The folder '{folder_path}' does not exist.")
        sys.exit(1)  # Exit if folder does not exist
else:
    print("Please provide a folder path as a command-line argument.")
    sys.exit(1)  # Exit if no argument is provided

audio_data_array = []

for file in wav_files:
    wav_file_path = os.path.join(folder_path, file)
    sample_rate, data = wavfile.read(wav_file_path)
    audio_data_array.append(data)

for audio in audio_data_array:
    print(f"Shape: {audio.shape}")
    print(f"Data Type: {audio.dtype}")
    print(f"Size: {audio.size}")
    print(f"Mean: {np.mean(audio)}")
    print(f"Standard Deviation: {np.std(audio)}")
    print(f"Min: {np.min(audio)}, Max: {np.max(audio)}\n")



for audio in audio_data_array:
    for amplitude in audio[:, 0]:
        if amplitude > Threshold:
          print(f"Value {amplitude} exceeds the threshold of {Threshold}" )


## load all files from a specified folder into a list of numpy arrays
def load_audio_files(folder_path):
    audio_data_array = []
    wav_files = [file for file in os.listdir(folder_path) if file.endswith('.wav')]
    for file in wav_files:
        wav_file_path = os.path.join(folder_path, file)
        sample_rate, data = wavfile.read(wav_file_path)
        audio_data_array.append(data)
    return audio_data_array

## show an histogram of the amplitude values of the audio files
def show_histogram(audio_data_array):
    for audio in audio_data_array:
        plt.hist(audio[:, 0], bins=50, alpha=0.7)
    plt.title('Histogram of Amplitude Values')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.show()