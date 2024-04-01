import os
import numpy as np
from scipy.io import wavfile
import argparse
from tqdm import tqdm  # Import tqdm for progress indication
import sys

class AudioProcessor:
    """
    Processes audio files to detect and isolate potential clicks based on amplitude threshold.
    
    Attributes:
        folder_path (str): Base folder path for processing.
        processed_folder (str): Subfolder name where processed files are located.
        potential_clicks_folder (str): Subfolder name for storing files with potential clicks isolated.
        threshold (float): Amplitude threshold for click detection.
        sample_rate (int): Sample rate of the audio files.
        peak_detect_ms (int): Time window in milliseconds for detecting the peak around a potential click.
    """
    
    def __init__(self, folder_path, processed_folder="processed", potential_clicks_folder="potential_clicks", threshold=0.2, sample_rate=44100, peak_detect_ms=30):
        """
        Initializes the audio processor with given directory paths and processing parameters.
        """
        self.folder_path = folder_path
        self.processed_folder = processed_folder
        self.potential_clicks_folder = potential_clicks_folder
        self.threshold = threshold  # Threshold for click detection
        self.sample_rate = sample_rate
        self.peak_detect_ms = peak_detect_ms  # Time window for peak detection in milliseconds

        # Compute the paths to the processing folders
        self.processed_files_path = os.path.join(self.folder_path, self.processed_folder)
        self.potential_clicks_path = os.path.join(self.folder_path, self.potential_clicks_folder)
        # Create the potential_clicks folder if it doesn't exist
        if not os.path.exists(self.potential_clicks_path):
            os.makedirs(self.potential_clicks_path)
    
    def normalize(self, signal):
        """
        Normalizes an audio signal so its maximum absolute value is 1.
        
        Parameters:
            signal (numpy.ndarray): The audio signal to normalize.
            
        Returns:
            numpy.ndarray: The normalized audio signal.
        """
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val != 0 else signal

    def detect_clicks_and_cut(self, data):
        """
        Detects potential clicks in the first channel of the audio data based on a threshold and isolates segments around them.
        
        Parameters:
            data (numpy.ndarray): The stereo audio data to analyze.
            
        Returns:
            list of tuples: Each tuple contains the index of the detected click and the isolated audio segment around it.
        """
        potential_clicks = []
        i = 0
        normalized_data = self.normalize(data[:, 0])  # Normalize the first channel
        
        while i < len(data):
            if normalized_data[i] > self.threshold:  # Analysis is based on the first channel
                start_index = i
                search_end_index = start_index + int(self.peak_detect_ms * self.sample_rate / 1000)
                highest_peak = 0
                highest_peak_index = i

                # Search for the highest peak within the specified time window
                while i <= search_end_index and i < len(data):
                    if normalized_data[i] > highest_peak:
                        highest_peak = normalized_data[i]
                        highest_peak_index = i
                    i += 1
                # Isolate the segment around the detected click
                potential_click, next_index = self.cutting(data, highest_peak_index)  # Cuts both channels
                potential_clicks.append((highest_peak_index, potential_click))
                i = next_index
            else:
                i += 1
                
        return potential_clicks
    
    def cutting(self, data, highest_peak_index):
        """"
        Isolates a segment around the detected click, including both audio channels.

        Parameters:
            data (numpy.ndarray): The stereo audio data.
            highest_peak_index (int): The index of the highest peak within the detected click segment.

        Returns:
            tuple: The first element is the isolated audio segment, and the second element is the end index of the segment.
        """
        samples_to_add = int(0.01 * self.sample_rate) # 10ms in samples
        start_cut = max(0, highest_peak_index - samples_to_add)
        end_cut = min(len(data), highest_peak_index + samples_to_add)
        potential_click = data[start_cut:end_cut]  # Includes both channels
        return potential_click, end_cut

    def process_files(self):
        """
        Processes all .wav files in the specified directory, detecting and isolating potential clicks.
        """
        wav_files = [file for file in os.listdir(self.processed_files_path) if file.endswith('.wav')]
        for file in tqdm(wav_files, desc="Processing files"):
            wav_file_path = os.path.join(self.processed_files_path, file)
            try:
                sample_rate, data = wavfile.read(wav_file_path)
                if data.shape[1] != 2:  # Checks if the file is stereo
                    print(f"File {file} is not stereo and will be skipped.", file=sys.stderr)
                    continue
                self.sample_rate = sample_rate

                potential_clicks = self.detect_clicks_and_cut(data)

                # Save isolated click segments
                for idx, (click_index, potential_click) in enumerate(potential_clicks):
                    click_file_path = os.path.join(self.potential_clicks_path, f"{file[:-4]}_click_{click_index}.wav")
                    wavfile.write(click_file_path, self.sample_rate, potential_click.astype(np.int16))  # Saves stereo data
            except Exception as e:
                print(f"Error processing file {file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioProcessor for detecting and cutting clicks in WAV files.")
    parser.add_argument('folder_path', type=str, help="The path to the folder containing the WAV files.")
    
    args = parser.parse_args()
    
    audio_processor = AudioProcessor(args.folder_path)
    audio_processor.process_files()
