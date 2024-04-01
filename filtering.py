import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from tqdm import tqdm
import sys

class AudioAnalyzer:
    def __init__(self, folder_path, potential_clicks_folder="potential_clicks"):
        """
        Initializes the audio analyzer with the specified folder paths.
        
        Parameters:
            folder_path (str): The base directory path where the audio files are located.
            potential_clicks_folder (str): The name of the subfolder within the base directory
                                            where potential clicks are stored.
        """
        self.folder_path = folder_path
        self.potential_clicks_folder = potential_clicks_folder
        self.sample_rate = None
        
        # Ensure the potential_clicks_path exists
        self.potential_clicks_path = os.path.join(self.folder_path, self.potential_clicks_folder)
        if not os.path.exists(self.potential_clicks_path):
            os.makedirs(self.potential_clicks_path)

    def normalize(self, signal):
        """
        Normalizes the audio signal so that its maximum absolute value is 1.
        
        Parameters:
            signal (numpy.ndarray): The audio signal to be normalized.
        
        Returns:
            numpy.ndarray: The normalized audio signal.
        """
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val != 0 else signal

    def analyze_offsets(self, data):
        """
        Analyzes the offset between the two stereo channels using cross-correlation.
        
        Parameters:
            data (numpy.ndarray): The stereo audio data to be analyzed.
        
        Returns:
            int: The calculated offset value.
        """
        correlation = correlate(self.normalize(data[:, 0]), self.normalize(data[:, 1]), mode='full')
        offset = correlation.argmax() - (len(correlation) // 2)
        return offset
    

    def process_files(self):
        """
        Processes all .wav files in the potential clicks directory, excluding those already marked with an offset.
        Renames the files to include the calculated offset in their filename.
        """
        # Exclude files already marked with an offset
        wav_files = [file for file in os.listdir(self.potential_clicks_path) if file.endswith('.wav') and "_offset_" not in file]
        for file in tqdm(wav_files, desc="Processing files"):
            wav_file_path = os.path.join(self.potential_clicks_path, file)
            try:
                sample_rate, data = wavfile.read(wav_file_path)
                if data.shape[1] != 2:  # Check if the file is stereo
                    print(f"File {file} is not stereo and will be skipped.", file=sys.stderr)
                    continue
                
                offset = self.analyze_offsets(data)
                # Rename the file to include the offset in the filename
                new_file_name = f"{file[:-4]}_offset_{offset}.wav"
                new_file_path = os.path.join(self.potential_clicks_path, new_file_name)
                os.rename(wav_file_path, new_file_path)
            except Exception as e:
                print(f"Error processing file {file}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        analyzer = AudioAnalyzer(folder_path)
        analyzer.process_files()
    else:
        print("Please provide the folder path as an argument.", file=sys.stderr)
