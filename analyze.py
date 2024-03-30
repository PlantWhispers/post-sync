import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

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
    audio, _ = librosa.load(os.path.join(folder_path, file), sr=None)
    audio_data_array.append(audio)

audio = np.concatenate(audio_data_array)

# Calculate the amplitude envelope
frame_length = 64
hop_length = 32  # These parameters can be adjusted
amplitude_envelope = np.max(librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length), axis=0)


plt.figure(figsize=(10, 6))
plt.hist(amplitude_envelope, bins=100)  # Adjust the number of bins for finer or coarser granularity
plt.title('Histogram of Amplitude Envelope')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.show()