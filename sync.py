import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, butter, filtfilt
from tqdm import tqdm

class AudioSyncFilter:
    """
    A class to process audio files for synchronization and filtering.
    
    Attributes:
        folder_path (str): Path to the directory containing WAV files to process.
        sync_duration_ms (int): Duration in milliseconds of the synchronization signal at the start of the audio.
    """
    
    def __init__(self, folder_path, sync_duration_ms=100):
        """
        Initialize the AudioSyncFilter class.
        """
        self.folder_path = folder_path
        self.sync_duration_ms = sync_duration_ms
        
        # Check if the given folder path exists, exit if it does not
        if not os.path.exists(self.folder_path):
            print(f"The folder '{self.folder_path}' does not exist.")
            sys.exit(1)

    def normalize(self, signal):
        """
        Normalize an audio signal so that its amplitude ranges between -1 and 1.
        
        Parameters:
            signal (numpy.ndarray): The audio signal to normalize.
            
        Returns:
            numpy.ndarray: The normalized audio signal.
        """
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val != 0 else signal

    def segment_cross_correlation(self, data):
        """
        Calculate the cross-correlation of two channels in a stereo audio segment.
        
        Parameters:
            data (numpy.ndarray): Stereo audio data.
            
        Returns:
            numpy.ndarray: Cross-correlation series.
        """
        # Calculate the number of samples for the given synchronization duration
        sync_length = int(self.sample_rate * self.sync_duration_ms / 1000)
        segment = data[:sync_length]  # Get the initial segment for synchronization
        correlations = correlate(self.normalize(segment[:, 0]), self.normalize(segment[:, 1]), mode='full')
        return correlations

    def find_offset(self, correlation):
        """
        Determine the offset between two audio channels from the cross-correlation.
        
        Parameters:
            correlation (numpy.ndarray): The cross-correlation array from segment_cross_correlation method.
            
        Returns:
            int: The offset in samples between the two audio channels.
        """
        # The peak of the cross-correlation corresponds to the offset
        offset = correlation.argmax() - (len(correlation) // 2)
        return offset

    def modify_data(self, data, offset):
        """
        Adjust audio data based on the offset to synchronize channels.
        
        Parameters:
            data (numpy.ndarray): The original stereo audio data.
            offset (int): The calculated offset from find_offset method.
            
        Returns:
            numpy.ndarray: The synchronized audio data.
        """
        # Shift data to synchronize the audio channels based on the offset
        if offset > 0:
            return np.column_stack((data[abs(offset):, 0], data[:-abs(offset), 1]))
        elif offset < 0:
            return np.column_stack((data[:-abs(offset), 0], data[abs(offset):, 1]))
        return data

    def high_pass_filter(self, data, sample_rate, cutoff=20000, order=5):
        """
        Apply a high-pass filter to the audio data.
        
        Parameters:
            data (numpy.ndarray): The audio data to filter.
            sample_rate (int): The sample rate of the audio data.
            cutoff (int): The cutoff frequency for the high-pass filter.
            order (int): The order of the filter.
            
        Returns:
            numpy.ndarray: The filtered audio data.
        """
        # Setup the high-pass filter
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data, axis=0)

    def process_wav_file(self, wav_filename):
        """
        Process a WAV file to synchronize its stereo channels and apply a high-pass filter.
        
        Parameters:
            wav_filename (str): Path to the WAV file.
            
        Returns:
            tuple: A tuple containing the processed audio data and the sample rate.
        """
        # Read the WAV file data and sample rate
        sample_rate, data = wavfile.read(wav_filename)
        # Ensure the file is stereo
        if data.shape[1] != 2:
            raise ValueError(f"WAV file {wav_filename} is not stereo")
        # Process the file for synchronization and filtering
        correlations = self.segment_cross_correlation(data)
        offset = self.find_offset(correlations)

        # Trim the sync signal and apply modifications
        data = data[int(sample_rate * self.sync_duration_ms / 1000):]
        new_data = self.modify_data(data, offset)
        filtered_data = self.high_pass_filter(new_data, sample_rate)

        return filtered_data, sample_rate

    def process_directory(self):
        """
        Process all WAV files in the specified directory.
        
        This method synchronizes the stereo channels of each WAV file and applies a high-pass filter,
        then saves the processed files in a new directory.
        """
        # List all WAV files in the directory
        wav_files = [file for file in os.listdir(self.folder_path) if file.endswith('.wav')]
        print(f"Found {len(wav_files)} WAV files for processing.")

        # Create a new directory to store the processed files
        new_directory = os.path.join(self.folder_path, "processed")
        os.makedirs(new_directory, exist_ok=True)

        # Process each file and save the result
        for file in tqdm(wav_files, desc="Processing WAV files", unit="file"):
            base_filename = file.rsplit('.', 1)[0]
            processed_file_path = os.path.join(new_directory, f"{base_filename}_processed.wav")

            # Skip processing if the file already exists
            if os.path.exists(processed_file_path):
                print(f"Skipping {file}, processed version already exists.")
                continue

            # Try processing the file and handle any errors
            try:
                wav_file_path = os.path.join(self.folder_path, file)
                new_data, sample_rate = self.process_wav_file(wav_file_path)
                wavfile.write(processed_file_path, sample_rate, new_data.astype(np.int16))
                print(f"Processed and saved {file} successfully.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        audio_sync_filter = AudioSyncFilter(folder_path)
        audio_sync_filter.process_directory()
    else:
        print("Usage: python script.py <path_to_directory>")
