import os
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np

class AudioEnvelopeAnalyzer:
    """
    A class to analyze the amplitude envelope of audio files within a directory.
    
    Attributes:
        folder_path (str): The directory containing the audio files.
        frame_length (int): The number of samples per frame for amplitude envelope calculation.
        hop_length (int): The number of samples to hop between frames.
        bins (int): The number of bins for the histogram plot.
        figsize (list): The size of the figure for the histogram plot.
    """

    def __init__(self, folder_path, frame_length=64, hop_length=32, bins=100, figsize=[10, 6]):
        """
        Initializes the AudioEnvelopeAnalyzer with the path to the directory of audio files.
        """
        self.folder_path = folder_path
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.bins = bins
        self.figsize = figsize
        
        # Verify that the folder path exists and contains WAV files
        if not os.path.exists(self.folder_path):
            print(f"The folder '{self.folder_path}' does not exist.")
            sys.exit(1)
        self.wav_files = [file for file in os.listdir(self.folder_path) if file.endswith('.wav')]
        if len(self.wav_files) == 0:
            print("No .wav files found in the specified folder.")
            sys.exit(1)

    def load_audio_data(self):
        """
        Load audio data from all .wav files in the directory.
        
        Returns:
            numpy.ndarray: Concatenated audio data from all files.
        """
        audio_data_array = []
        for file in self.wav_files:
            audio, _ = librosa.load(os.path.join(self.folder_path, file), sr=None)
            audio_data_array.append(audio)
        return np.concatenate(audio_data_array)

    def calculate_amplitude_envelope(self, audio):
        """
        Calculate the amplitude envelope of the audio signal.
        
        Parameters:
            audio (numpy.ndarray): The audio data to calculate the amplitude envelope.
            
        Returns:
            numpy.ndarray: The amplitude envelope of the audio.
        """
        return np.max(librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length), axis=0)

    def plot_amplitude_envelope_histogram(self, amplitude_envelope):
        """
        Plot a histogram of the amplitude envelope.
        
        Parameters:
            amplitude_envelope (numpy.ndarray): The amplitude envelope to plot.
        """
        plt.figure(figsize=self.figsize)
        plt.hist(amplitude_envelope, bins=self.bins)
        plt.title('Histogram of Amplitude Envelope')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.show()

    def analyze(self):
        """
        Perform the complete analysis by loading the audio, calculating the amplitude envelope,
        and plotting the histogram.
        """
        audio_data = self.load_audio_data()
        amplitude_envelope = self.calculate_amplitude_envelope(audio_data)
        self.plot_amplitude_envelope_histogram(amplitude_envelope)

if __name__ == "__main__":
    # Accepts folder path as a command-line argument to initiate the analysis
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        analyzer = AudioEnvelopeAnalyzer(folder_path)
        analyzer.analyze()
    else:
        print("Please provide a folder path as a command-line argument.")
        sys.exit(1)
