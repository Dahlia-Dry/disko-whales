"""
Disko_Sound: Acoustic feature extraction and analysis for whale sound classification
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')


class Disko_Sound:
    """
    A class for extracting and analyzing acoustic features from whale sound .wav files.
    
    Features extracted include:
    - Spectrograms and visualizations
    - Frequency domain: dominant frequency, bandwidth, spectral centroid, spectral rolloff
    - Temporal: call duration, inter-call intervals, rhythm patterns
    - Energy: signal-to-noise ratio, amplitude envelope
    - Advanced: MFCCs (Mel-frequency cepstral coefficients)
    """
    
    def __init__(self, wav_file_path, sr=None):
        """
        Initialize Disko_Sound with a .wav file.
        
        Parameters
        ----------
        wav_file_path : str
            Path to the .wav file
        sr : int, optional
            Sample rate. If None, uses the file's native sample rate
        """
        self.wav_file_path = wav_file_path
        self.y, self.sr = librosa.load(wav_file_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.n_samples = len(self.y)
        
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def plot_spectrogram(self, figsize=(14, 6), cmap='viridis', vmin=None, vmax=None,
                         freq_range=None, title=None):
        """
        Create and display a spectrogram of the audio signal.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap to use
        vmin, vmax : float, optional
            Min/max values for color scaling
        freq_range : tuple, optional
            Frequency range to display (min_freq, max_freq) in Hz
        title : str, optional
            Title for the plot
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        # Compute Short-Time Fourier Transform
        D = librosa.stft(self.y)
        S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display spectrogram
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz',
                                       ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Set frequency range if specified
        if freq_range:
            ax.set_ylim(freq_range)
        
        ax.set_title(title or f'Spectrogram: {self.wav_file_path}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return fig, ax
    
    def plot_waveform(self, figsize=(14, 4)):
        """
        Plot the waveform of the audio signal.
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        times = np.arange(len(self.y)) / self.sr
        ax.plot(times, self.y, lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform: {self.wav_file_path}')
        ax.grid(True, alpha=0.3)
        return fig, ax
    
    def plot_spectrogram_and_waveform(self, figsize=(14, 10), cmap='viridis'):
        """
        Create a figure with both waveform and spectrogram.
        
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Waveform
        times = np.arange(len(self.y)) / self.sr
        axes[0].plot(times, self.y, lw=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Waveform: {self.wav_file_path}')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.stft(self.y)
        S_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz',
                                       ax=axes[1], cmap=cmap)
        axes[1].set_title('Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        return fig, axes
    
    # ============================================================================
    # FREQUENCY DOMAIN FEATURES
    # ============================================================================
    
    def get_dominant_frequency(self, freq_range=None):
        """
        Extract the dominant frequency (peak frequency) of the signal.
        
        Parameters
        ----------
        freq_range : tuple, optional
            Frequency range to search within (min_freq, max_freq)
        
        Returns
        -------
        float
            Dominant frequency in Hz
        """
        # Compute power spectrum
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Apply frequency range if specified
        if freq_range:
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            mean_power = mean_power * mask
        
        # Find peak
        dominant_freq = freqs[np.argmax(mean_power)]
        return float(dominant_freq)
    
    def get_spectral_centroid(self):
        """
        Extract the spectral centroid (center of mass of the spectrum).
        
        Returns
        -------
        float
            Mean spectral centroid in Hz
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        return float(np.mean(spectral_centroids))
    
    def get_spectral_rolloff(self, percent=0.95):
        """
        Extract the spectral rolloff frequency (frequency below which 85% of energy is contained).
        
        Parameters
        ----------
        percent : float
            Energy percent threshold (default 0.95 = 95%)
        
        Returns
        -------
        float
            Mean spectral rolloff frequency in Hz
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr,
                                                            roll_percent=percent)[0]
        return float(np.mean(spectral_rolloff))
    
    def get_bandwidth(self, freq_range=None):
        """
        Extract the bandwidth of the signal.
        
        Parameters
        ----------
        freq_range : tuple, optional
            Frequency range to consider
        
        Returns
        -------
        float
            Bandwidth in Hz (defined as rolloff - lowest significant freq)
        """
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find frequencies with significant energy (above threshold)
        threshold = np.max(mean_power) * 0.05  # 5% threshold
        significant_freqs = freqs[mean_power > threshold]
        
        if len(significant_freqs) > 0:
            bandwidth = float(np.max(significant_freqs) - np.min(significant_freqs))
        else:
            bandwidth = 0.0
        
        return bandwidth
    
    def get_frequency_range(self):
        """
        Get the frequency range containing significant energy.
        
        Returns
        -------
        tuple
            (min_frequency, max_frequency) in Hz
        """
        D = librosa.stft(self.y)
        power_spectrum = np.abs(D) ** 2
        mean_power = np.mean(power_spectrum, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find frequencies with significant energy
        threshold = np.max(mean_power) * 0.05
        significant_freqs = freqs[mean_power > threshold]
        
        if len(significant_freqs) > 0:
            return float(np.min(significant_freqs)), float(np.max(significant_freqs))
        else:
            return 0.0, 0.0
    
    # ============================================================================
    # TEMPORAL FEATURES
    # ============================================================================
    
    def get_call_duration(self):
        """
        Get the total duration of the audio signal.
        
        Returns
        -------
        float
            Duration in seconds
        """
        return self.duration
    
    # ============================================================================
    # ENERGY FEATURES
    # ============================================================================
    
    def get_rms_energy(self):
        """
        Extract RMS (Root Mean Square) energy of the signal.
        
        Returns
        -------
        float
            RMS energy
        """
        rms = librosa.feature.rms(y=self.y)[0]
        return float(np.mean(rms))
    
    def get_energy_envelope(self):
        """
        Extract the amplitude envelope of the signal.
        
        Returns
        -------
        ndarray
            Normalized amplitude envelope
        """
        # Compute RMS energy in frames
        rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        
        # Normalize
        rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
        
        return rms_normalized
    
    def get_signal_to_noise_ratio(self, noise_duration=0.5):
        """
        Estimate Signal-to-Noise Ratio (SNR).
        
        Assumes the beginning of the signal contains noise.
        
        Parameters
        ----------
        noise_duration : float
            Duration in seconds of the signal assumed to be noise (default: first 0.5s)
        
        Returns
        -------
        float
            SNR in dB
        """
        # Extract noise from beginning
        noise_samples = int(noise_duration * self.sr)
        noise = self.y[:noise_samples]
        noise_power = np.mean(noise ** 2)
        
        # Get signal power (excluding noise region)
        signal = self.y[noise_samples:]
        signal_power = np.mean(signal ** 2)
        
        # Calculate SNR
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
        
        return float(snr_db)
    
    def get_zero_crossing_rate(self):
        """
        Extract Zero Crossing Rate (ZCR) - useful for distinguishing voiced/unvoiced segments.
        
        Returns
        -------
        float
            Mean zero crossing rate
        """
        zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        return float(np.mean(zcr))
    
    